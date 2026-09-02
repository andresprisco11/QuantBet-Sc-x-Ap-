# Registra la actualizacion automatica en el Programador de tareas de Windows.
# Correr UNA vez, como administrador.
#
# Por que Programador de tareas y no un cron de Python: corre aunque no
# tengas VS Code abierto, sobrevive a reinicios, y no depende de que haya
# una sesion de nadie encendida.
$raiz   = "C:\Users\Pipe\Proyecto Startup_Apuestas_IA"
$script = Join-Path $raiz "actualizar.ps1"
$nombre = "QuantBet - actualizacion"

$accion = New-ScheduledTaskAction -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`"" `
  -WorkingDirectory $raiz

# cada 3 horas: suficiente para que las cuotas no queden viejas y
# ~150 creditos por corrida, unas 1200 al dia sobre 19.000 disponibles
$disparador = New-ScheduledTaskTrigger -Once -At (Get-Date).Date.AddHours(7) `
  -RepetitionInterval (New-TimeSpan -Hours 3)

$config = New-ScheduledTaskSettingsSet -StartWhenAvailable `
  -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
  -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 25)

Register-ScheduledTask -TaskName $nombre -Action $accion -Trigger $disparador `
  -Settings $config -Description "Actualiza cuotas, resultados, noticias y fotos del mercado" -Force

Write-Host ""
Write-Host "Tarea '$nombre' registrada: cada 3 horas desde las 7:00."
Write-Host "Log: $raiz\data\runs\actualizacion.log"
Write-Host ""
Write-Host "Para correrla ya mismo:      Start-ScheduledTask -TaskName '$nombre'"
Write-Host "Para ver cuando corrio:      Get-ScheduledTaskInfo -TaskName '$nombre'"
Write-Host "Para apagarla:               Disable-ScheduledTask -TaskName '$nombre'"
