param(
    [string]$source = "C:\Repositories\eva-next-node\src\_course",
    [string]$dest = "C:\_Gen\_in\text",
    [string]$Filter = "*_text.json"
)

Get-ChildItem $source -Recurse -Filter $Filter | ForEach-Object {
    $relative = $_.FullName.Substring($source.Length).TrimStart('\')
    $target = Join-Path $dest $relative

    New-Item -ItemType Directory -Path (Split-Path $target) -Force | Out-Null
    Copy-Item $_.FullName $target -Force
}