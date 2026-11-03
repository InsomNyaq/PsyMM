# Release checklist

- [x] Local code-only review candidate assembled; no data or weights copied into the Git repository.
- [x] Original source snapshot hashes verified at collection; candidate scanned for obvious credential patterns.
- [x] Owner authorized the initial code-only publication; CPU tests and component import checks passed.
- [ ] Rotate credentials in excluded historical server scripts.
- [ ] Pin dependency and upstream component revisions; test setup on Linux.
- [ ] Verify complete audio and video generation in a clean Linux environment with authorized assets; the included CPU tests do not establish model output quality.
- [ ] Freeze representative sampling across source, language, emotion and persona, including failures.
- [ ] Confirm source/reference redistribution rights and privacy clearance per sample.
- [ ] Recheck Chinese generated audio before including it in a release.
- [ ] Transform absolute paths into relative public paths; remove private identifiers.
- [ ] Produce file hashes, schema, split/quality statistics and finalized data card.
- [ ] Create the authorized Hugging Face repository, upload in resumable batches and verify assets.
- [ ] Give reviewers a working public or reviewer-only access path and cite its exact version in the response.

The release scope, namespace and data terms need an owner decision. Reviewer access is not yet complete.
