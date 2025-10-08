
See **docs/EVALUATION.md** for the evaluation & picks pipeline details.



<!-- QUICK_START_BEGIN -->
## Hýzlý Baþlangýç

Aþaðýdaki zinciri çalýþtýrarak tüm deðerlendirme akýþýný tek seferde çalýþtýrabilirsiniz:

\\\powershell
python -m src.models.loo_eval; 
python -m tools.ensure_labels; 
python -m tools.make_picks; 
python -m tools.plot_eval; 
Start-Process .\reports\confusion_matrix.png; 
Start-Process .\reports\reliability_maxproba.png; 
Start-Process .\reports\loo_summary.json
\\\

Daha fazla ayrýntý için **docs/EVALUATION.md** dosyasýna bakýn.
<!-- QUICK_START_END -->
