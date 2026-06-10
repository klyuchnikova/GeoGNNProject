# Merge into `GeoGNNProject/flashback-branch`

This ZIP is an **overlay**. Extract its contents into the root of your checked-out `flashback-branch` and allow new files to be added. It intentionally does not overwrite these existing research artifacts:

- `GeoGNNProject.pdf`
- `foursquaregraphs-eda.ipynb`
- `gowalla_validated_metadata_eda.ipynb`

After extraction, verify:

```bash
python scripts/verify_merge.py
pytest -q
```

Commit the overlay together with the existing EDA files. `README_OLD.md` is retained only as a migration reference and may be deleted before the final commit.
