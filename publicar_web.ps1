# Copia la app a docs/ , que es lo que GitHub Pages publica.
# Correr despues de cada ciclo de actualizacion para que el telefono vea
# los datos nuevos, y hacer push.
$raiz = "C:\\Users\\Pipe\\Proyecto Startup_Apuestas_IA"
$destino = Join-Path $raiz "docs"
New-Item -ItemType Directory -Force -Path $destino | Out-Null
Copy-Item (Join-Path $raiz "app\\pwa\\*") $destino -Force
Copy-Item (Join-Path $raiz "app\\data.js")  $destino -Force
Copy-Item (Join-Path $raiz "app\\colors.js") $destino -Force
Write-Host "Listo. Ahora:"
Write-Host "   git add -A ; git commit -m 'actualizar app web' ; git push"
