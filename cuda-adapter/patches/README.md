# Patches

Numbered unified diffs (`NNNN-name.patch`) for any imported tree the adapter has to change, each with an
in-source comment saying what changed and why, applied to a COPY in `build/` and never to the original.
None are needed yet: the adapter replaces the library boundary, and the hybrid_ops client is built from
its unmodified source with renamed symbols plus `shim/hybrid_ops_wrapper_abi.c`.
