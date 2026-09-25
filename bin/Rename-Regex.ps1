param(
    [string]$Regex,
    [string]$Replace,
    [string]$Path = "."
)

Get-ChildItem -Path $Path -File | ForEach-Object {

    $name = $_.BaseName
    $ext = $_.Extension

    if ($name -match $Regex) {

        #TODO Needs implementation
        $number = $matches[1]
        $rest = $matches[2]

        $newName = "$rest" + "_" + "$number" + "$ext"
        Write-Host "'$($_.Name)' -> '$newName'"
        Rename-Item -Path $_.FullName -NewName $newName
    }
}