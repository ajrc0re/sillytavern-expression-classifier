[CmdletBinding()]
param ()
try {

    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
    $wingetOutput = winget list --exact --id ezwinports.make 2>&1

    $filteredOutput = $wingetOutput |
        ForEach-Object { $_.Trim() } |
        Where-Object {
            $_ -ne '' -and
            $_ -notmatch '^\w*\[\s*\d+/\d+\s*\]' -and
            $_ -notmatch '^\s*Found\s+\d+\s+packages\s*$' -and
            $_ -notmatch '^-+$' -and
            $_ -ne '-' -and
            $_ -ne '\'
        }

    $filteredOutput | ForEach-Object { Write-Output $_ }

    if ($filteredOutput -contains 'No installed package found matching input criteria.') {
        Write-Host 'Package not found, installing...'
        winget install --exact --id ezwinports.make
        exit 0
    } else {
        Write-Host 'Make is already installed.'
        exit 0
    }
} catch {
    $PSItem.InvocationInfo
    $PSCmdlet.ThrowTerminatingError($PSItem)
}
