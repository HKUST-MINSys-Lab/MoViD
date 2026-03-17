# NTU Transfer Notes

If you already downloaded NTU archives on another machine, copy them into:

- `dataset/NTU/NTU60/nturgbd_skeletons_s001_to_s017.zip`
- `dataset/NTU/NTU120/nturgbd_skeletons_s018_to_s032.zip`

Then run:

```bash
bash scripts/setup/setup_ntu_files.sh
```

Common transfer options:

- `scp /local/file.zip <user>@<server>:/path/to/MoViD/dataset/NTU/...`
- WinSCP / SFTP
- temporary local HTTP server plus `wget`
