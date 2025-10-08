
See **docs/EVALUATION.md** for the evaluation & picks pipeline details.



<!-- QUICK_START_BEGIN -->
## H�zl� Ba�lang��

A�a��daki zinciri �al��t�rarak t�m de�erlendirme ak���n� tek seferde �al��t�rabilirsiniz:

\\\powershell
python -m src.models.loo_eval; 
python -m tools.ensure_labels; 
python -m tools.make_picks; 
python -m tools.plot_eval; 
Start-Process .\reports\confusion_matrix.png; 
Start-Process .\reports\reliability_maxproba.png; 
Start-Process .\reports\loo_summary.json
\\\

Daha fazla ayr�nt� i�in **docs/EVALUATION.md** dosyas�na bak�n.
<!-- QUICK_START_END -->
