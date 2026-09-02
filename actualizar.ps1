# ===================================================================
#  QuantBet -- ciclo completo de actualizacion
#  Pensado para el Programador de tareas de Windows: no pide nada,
#  no abre ventanas, y deja registro de cada corrida en un log.
#
#  Orden deliberado:
#    1. archivar  primero, para que los partidos ya jugados salgan de
#                 data.js CON su precio previo antes de que se pisen
#    2. exportar  los partidos nuevos (fusiona, no reemplaza)
#    3. enriquecer con historial, forma y noticias
#    4. noticias  al final: no gasta creditos de la API de cuotas
# ===================================================================
$ErrorActionPreference = "Continue"
$raiz = "C:\Users\Pipe\Proyecto Startup_Apuestas_IA"
$log  = Join-Path $raiz "data\runs\actualizacion.log"
Set-Location $raiz
& "$raiz\venv\Scripts\Activate.ps1"

function Paso($nombre, $args) {
  $t = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  Add-Content $log "[$t] --- $nombre ---"
  $salida = & python @args 2>&1 | Out-String
  Add-Content $log $salida
  if ($LASTEXITCODE -ne 0) { Add-Content $log "[$t] $nombre FALLO ($LASTEXITCODE)" }
}

Paso "archivar resultados" @("-m","src.app.results_archive","--actualizar")
Paso "exportar cuotas"     @("-m","src.app.export_app_data","--ligas","top5,latam,soccer_japan_j_league")
Paso "enriquecer"          @("-m","src.app.enrich_app_data")
Paso "noticias"            @("-m","src.news.news_loader","--fetch")
Paso "foto del mercado"    @("-m","src.discovery.odds_snapshots","--capturar","--ligas","top5")
Paso "seleccion del dia"   @("-m","src.app.daily_slate","--generar")

$t = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content $log "[$t] === ciclo completo ===`n"
