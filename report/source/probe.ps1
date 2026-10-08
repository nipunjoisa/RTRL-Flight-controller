# Report the page each numbered heading / figure caption lands on, using
# Word's own pagination -- avoids the ExportAsFixedFormat hang entirely.
$dir  = $PSScriptRoot
$docx = Join-Path $dir "RTRL_Flight_Controller_Report.docx"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
  $doc = $word.Documents.Open($docx, [ref]$false, [ref]$true)
  Write-Output ("TOTAL_PAGES=" + $doc.ComputeStatistics(2))
  $sec2start = $doc.Sections.Item(2).Range.Start
  foreach ($p in $doc.Paragraphs) {
    $txt = $p.Range.Text.Trim()
    if ($txt -match '^(\d+\.\s|Figure \d+:|Table \d+:)') {
      $abs = $p.Range.Information(3)   # wdActiveEndPageNumber
      $rel = if ($p.Range.Start -ge $sec2start) { $abs - 3 } else { $abs }
      Write-Output ("PAGE " + $rel + " (abs " + $abs + ") :: " + $txt.Substring(0, [Math]::Min(70, $txt.Length)))
    }
  }
  $doc.Close([ref]$false)
} finally { $word.Quit() }
