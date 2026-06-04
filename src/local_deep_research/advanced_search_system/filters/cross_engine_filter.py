"""
Cross-engine search result filter implementation.
"""

from typing import Dict, List

from loguru import logger

from ...utilities.json_utils import extract_json, get_llm_response_text
from .base_filter import BaseFilter

# Per-preview snippet character cap. Kept in sync with the per-engine relevance
# filter (web_search_engines/relevance_filter.py:_SNIPPET_CHAR_CAP). 200 was too
# tight — it truncated academic abstracts before the ranker could judge whether a
# paper's primary topic matched the query, letting off-topic results rank highly.
_SNIPPET_CHAR_CAP = 800


class CrossEngineFilter(BaseFilter):
    """Filter that ranks and filters results from multiple search engines."""

    def __init__(
        self,
        model,
        max_results=None,
        default_reorder=True,
        default_reindex=True,
        settings_snapshot=None,
    ):
        """
        Initialize the cross-engine filter.

        Args:
            model: Language model to use for relevance assessment
            max_results: Maximum number of results to keep after filtering
            default_reorder: Default setting for reordering results by relevance
            default_reindex: Default setting for reindexing results after filtering
            settings_snapshot: Settings snapshot for thread context
        """
        super().__init__(model)
        # Import from thread_settings to avoid database dependencies
        from ...config.thread_settings import (
            get_setting_from_snapshot,
            NoSettingsContextError,
        )

        # Get max_results from database settings if not provided
        if max_results is None:
            try:
                max_results = get_setting_from_snapshot(
                    "search.cross_engine_max_results",
                    default=100,
                    settings_snapshot=settings_snapshot,
                )
                if max_results is not None:
                    max_results = int(max_results)
                else:
                    max_results = 100
            except (NoSettingsContextError, TypeError, ValueError):
                max_results = 100
        self.max_results = max_results

        # Max number of result previews shown to the LLM for relevance ranking.
        # Higher values let the LLM evaluate more candidates but increase prompt
        # size and latency.
        try:
            self.max_context_items = int(
                get_setting_from_snapshot(
                    "search.cross_engine_max_context_items",
                    default=30,
                    settings_snapshot=settings_snapshot,
                )
            )
        except (NoSettingsContextError, TypeError, ValueError):
            self.max_context_items = 30

        self.default_reorder = default_reorder
        self.default_reindex = default_reindex

    def _prepare_and_return(self, results, *, reindex, start_index):
        """Optionally reindex results and return them."""
        if reindex:
            for i, result in enumerate(results):
                result["index"] = str(i + start_index + 1)
        return results

    def _build_rank_prompt(self, batch, query):
        """Build the relevance-ranking prompt for one batch of results.

        ``batch`` is a list of ``(global_index, result)`` pairs. Previews are
        numbered **locally** (0-based) within the batch so the LLM never has to
        reason about large or sparse indices; callers map the returned local
        indices back to global ones.
        """
        preview_context = []
        for local_idx, (_global_idx, result) in enumerate(batch):
            title = result.get("title", "Untitled").strip()
            snippet = result.get("snippet", "").strip()
            engine = result.get("engine", "Unknown engine")

            # Clean up snippet if too long
            if len(snippet) > _SNIPPET_CHAR_CAP:
                snippet = snippet[:_SNIPPET_CHAR_CAP] + "..."

            preview_context.append(
                f"[{local_idx}] Engine: {engine} | Title: {title}\nSnippet: {snippet}"
            )

        context = "\n\n".join(preview_context)

        return f"""You are a search result filter. Your task is to rank search results from multiple engines by relevance to a query.

Query: "{query}"

Search Results:
{context}

Return the search results as a JSON array of indices, ranked from most to least relevant to the query.
Only include indices of results that are actually relevant to the query.
For example: [3, 0, 7, 1]

If no results seem relevant to the query, return an empty array: []"""

    def _rank_batch(self, batch, query):
        """Rank a single batch via the LLM.

        Returns a list of **global** indices ordered most→least relevant.
        Returns an empty list when the model parsed a response but kept nothing
        valid (distinct from the all-filtered case the caller handles), and
        ``None`` when the response could not be parsed or the call raised — so
        the caller can tell "rejected all" apart from "filter unavailable".
        """
        prompt = self._build_rank_prompt(batch, query)
        try:
            response = self.model.invoke(prompt)
            response_text = get_llm_response_text(response)
            ranked_local = extract_json(response_text, expected_type=list)
        except Exception:
            logger.exception("Cross-engine batch ranking error")
            return None

        if ranked_local is None:
            return None

        ordered = []
        for local_idx in ranked_local:
            # bool is an int subclass — reject True/False explicitly.
            if isinstance(local_idx, bool):
                continue
            if isinstance(local_idx, int) and 0 <= local_idx < len(batch):
                ordered.append(batch[local_idx][0])
        return ordered

    def _rank_indices(self, results, query, effective_max):
        """Return global indices of ``results`` ordered most→least relevant.

        A pool that fits in a single context window (``len <=
        max_context_items``) issues exactly one LLM call, preserving historical
        behavior. Larger pools are split into ``max_context_items`` chunks,
        ranked independently, and round-robin merged so the best of every batch
        surfaces into the head — no candidate beyond the first chunk is silently
        dropped, which was the previous behavior.

        Returns ``None`` only when *every* batch failed to parse (caller falls
        back to the capped unranked slice); ``[]`` when batches parsed but kept
        nothing (caller falls back to the top-10 originals).
        """
        # Bound LLM cost: never rank more candidates than we could keep, but
        # always consider at least one full context window.
        candidate_cap = max(effective_max, self.max_context_items)
        candidates = list(enumerate(results))[:candidate_cap]

        batch_size = max(1, self.max_context_items)
        batches = [
            candidates[i : i + batch_size]
            for i in range(0, len(candidates), batch_size)
        ]

        per_batch_orders = []
        any_parsed = False
        for batch in batches:
            order = self._rank_batch(batch, query)
            if order is None:
                continue
            any_parsed = True
            if order:
                per_batch_orders.append(order)

        if not any_parsed:
            return None
        if not per_batch_orders:
            return []
        if len(per_batch_orders) == 1:
            return per_batch_orders[0]

        # Round-robin merge: take each batch's top pick, then each batch's
        # second pick, and so on. Surfaces the best of every batch without
        # needing cross-batch score calibration.
        merged = []
        longest = max(len(order) for order in per_batch_orders)
        for rank in range(longest):
            for order in per_batch_orders:
                if rank < len(order):
                    merged.append(order[rank])
        return merged

    def filter_results(
        self,
        results: List[Dict],
        query: str,
        reorder=None,
        reindex=None,
        start_index=0,
        max_results=None,
        **kwargs,
    ) -> List[Dict]:
        """
        Filter and rank search results from multiple engines by relevance.

        Args:
            results: Combined list of search results from all engines
            query: The original search query
            reorder: Whether to reorder results by relevance (default: use instance default)
            reindex: Whether to update result indices after filtering (default: use instance default)
            start_index: Starting index for the results (used for continuous indexing)
            max_results: Per-call override for the maximum number of results to
                keep. Falls back to the instance ``max_results`` when None.
            **kwargs: Additional parameters

        Returns:
            Filtered list of search results
        """
        # Use instance defaults if not specified
        if reorder is None:
            reorder = self.default_reorder
        if reindex is None:
            reindex = self.default_reindex

        effective_max = (
            self.max_results if max_results is None else int(max_results)
        )

        if not self.model or len(results) <= 10:  # Don't filter if few results
            return self._prepare_and_return(
                results[: min(effective_max, len(results))],
                reindex=reindex,
                start_index=start_index,
            )

        ranked_indices = self._rank_indices(results, query, effective_max)

        if ranked_indices is None:
            logger.info(
                "Cross-engine filtering could not rank results, returning capped originals"
            )
            return self._prepare_and_return(
                results[: min(effective_max, len(results))],
                reindex=reindex,
                start_index=start_index,
            )

        # If not reordering, just filter based on the indices (keep originals'
        # relative order).
        if not reorder:
            filtered_results = [results[idx] for idx in sorted(ranked_indices)]
            final_results = filtered_results[
                : min(effective_max, len(filtered_results))
            ]

            if not final_results and results:
                logger.info(
                    "Cross-engine filtering removed all "
                    "results, returning top 10 originals"
                )
                return self._prepare_and_return(
                    results[: min(10, len(results))],
                    reindex=reindex,
                    start_index=start_index,
                )

            logger.info(
                f"Cross-engine filtering kept {len(final_results)} out of {len(results)} results without reordering"
            )
            return self._prepare_and_return(
                final_results,
                reindex=reindex,
                start_index=start_index,
            )

        # Create ranked results list (reordering)
        ranked_results = [results[idx] for idx in ranked_indices]

        # If filtering removed everything, return top results
        if not ranked_results and results:
            logger.info(
                "Cross-engine filtering removed all results, returning top 10 originals instead"
            )
            return self._prepare_and_return(
                results[: min(10, len(results))],
                reindex=reindex,
                start_index=start_index,
            )

        # Limit results if needed
        final_results = ranked_results[: min(effective_max, len(ranked_results))]

        logger.info(
            f"Cross-engine filtering kept {len(final_results)} out of {len(results)} results with reordering={reorder}, reindex={reindex}"
        )
        return self._prepare_and_return(
            final_results,
            reindex=reindex,
            start_index=start_index,
        )
