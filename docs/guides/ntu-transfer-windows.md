# NTU Transfer from Windows

## Recommended: SCP from PowerShell

```powershell
scp D:\download\nturgbd_skeletons_s018_to_s032.zip <user>@<server>:/path/to/MoViD/dataset/NTU/NTU120/
scp D:\download\nturgbd_skeletons_s001_to_s017.zip <user>@<server>:/path/to/MoViD/dataset/NTU/NTU60/
```

## After upload

Run on the Linux machine:

```bash
cd /path/to/MoViD
bash scripts/setup/setup_ntu_files.sh
```

You can also use WinSCP or any SFTP client if you prefer a GUI workflow.
