# STM32 Service — navodila za Windows

### 1. Namesti NSSM

Prenesi z https://nssm.cc/download — razpakiraš in daš `nssm.exe` npr. v `C:\Tools\` (win64 verzija za večino).

### 2. Registriraj service (PowerShell kot Administrator)

```powershell
C:\Tools\nssm.exe install STM32Service "<interpreter_path>" "C:\pot\do\stm_server.py"
C:\Tools\nssm.exe set STM32Service AppDirectory "C:\pot\do\"
C:\Tools\nssm.exe set STM32Service DisplayName "SPO STM32 Service"
C:\Tools\nssm.exe set STM32Service Description "STM32 TCP/IP bridge service"
C:\Tools\nssm.exe set STM32Service Start SERVICE_AUTO_START
```

<interpreter_path> je pot do interpreterja, ki vam bo zagnal server npr.: C:\Python312\python.exe

STM32Service je samo primer imena, lahko damo kar želimo

Lahko tudi z GUI (C:\Tools\nssm.exe install STM32Service) in kjer se odpre okno in tam vnesemo vse potrebno

### 3. Upravljanje

```powershell
# Zagon
nssm start STM32Service

# Status
nssm status STM32Service

# Ustavitev
nssm stop STM32Service

# Odstranitev
nssm remove STM32Service confirm
```

### 4. Preverjanje logov

Logi so v `stm_server.log` poleg `stm_server.py`, ali jih nastavi z NSSM:

```powershell
nssm set STM32Service AppStdout "C:\pot\do\service_stdout.log"
nssm set STM32Service AppStderr "C:\pot\do\service_stderr.log"
```

---

## Testiranje storitve

```powershell
# Poveži se s storitvijo (Windows — ncat iz nmap paketa)
ncat 127.0.0.1 5000

# Ali z vgrajenim telnetom
telnet 127.0.0.1 5000
```

### Primeri ukazov v odprtem ncat/telnet oknu:

```
STATUS
GET_LAST
GET_ALL
GET_FILE|data.bin
DELETE
```
