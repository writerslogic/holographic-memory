# MongoDB / Atlas Vector Search

`hms-import mongodb --uri "$MONGODB_URI" --db app --collection memories --vector-field embedding --text-field text`

- Reads documents with `find().sort(_id).limit(page)` and `_id > last`, so it is resumable (the cursor encodes ObjectId, string or integer `_id`). Other `_id` types are rejected with a reason.
- Atlas Vector Search indexes are not needed or queried: the importer only reads the stored vector field (array or BSON vector). `--filter` takes a JSON query, `--metadata-fields` limits metadata, `--namespace-field`, `--created-field`, `--updated-field` map those fields. ObjectId and other BSON scalars become strings; non-finite floats are kept as strings and counted.
- URI (`mongodb+srv://` supported) from `--uri` or `$MONGODB_URI`; the embedded password is registered as a secret and redacted everywhere.

Verification: MOCK. No mongod is available; the tests use an in-process fake of the exact `find/sort/limit/count_documents` surface used. pymongo against a real server and Atlas itself are unverified.
