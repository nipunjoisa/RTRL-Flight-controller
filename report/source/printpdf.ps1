# ExportAsFixedFormat hangs in this environment (even on an empty document),
# so drive the "Microsoft Print to PDF" queue instead -- a different code path.
# Late-bound via InvokeMember: PS 5.1 mangles [ref]<string> for COM calls.
$dir  = $PSScriptRoot
$docx = Join-Path $dir "RTRL_Flight_Controller_Report.docx"
$pdf  = [string](Join-Path $dir "RTRL_Flight_Controller_Report.pdf")
if (Test-Path $pdf) { Remove-Item $pdf -Force }
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
  $doc = $word.Documents.Open($docx, [ref]$false, [ref]$true)
  $word.ActivePrinter = "Microsoft Print to PDF"
  $word.Options.PrintBackground = $false
  Write-Output ("printing " + $doc.ComputeStatistics(2) + " pages")
  # Background, Append, Range, OutputFileName, From, To, Item, Copies, Pages,
  # PageType, PrintToFile
  $args = @($false, $false, 0, $pdf, [Type]::Missing, [Type]::Missing, 0, 1,
            [Type]::Missing, 0, $true)
  [void]$doc.GetType().InvokeMember("PrintOut", "InvokeMethod", $null, $doc, $args)
  Write-Output "printout returned"
  $doc.Close([ref]$false)
} finally { $word.Quit() }
for ($i = 0; $i -lt 60; $i++) {
  if (Test-Path $pdf) { break }
  Start-Sleep -Milliseconds 500
}
if (Test-Path $pdf) { Write-Output ("OK " + (Get-Item $pdf).Length) } else { Write-Output "MISSING" }
