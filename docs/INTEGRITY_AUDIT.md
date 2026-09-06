# Superseded perfect-score archive audit

Audited archive SHA-256: `f4f5dbdf20733ca27cb99e23debe7773abf985945de7e9e8decf45175d610950`

The archive's root submission has SHA-256 `7336e3992078b3cbe68e275dbefe262e7a1e2b44eded7d9e34c9b89b8991bf88` and contains only hard `0/1` predictions. Its own documentation and code show that 5,991 test labels were copied from an external UCI source table and 9 were resolved by subtracting labelled training-row counts. The model fallback was used for zero rows.

This is label reconstruction, not evaluation on genuinely unseen outcomes. The method and all external source-label assets are excluded from the final repository. The audit finding is retained to make the version transition transparent; the original ZIP itself is not committed.
