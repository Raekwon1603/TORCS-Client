Voor de data pipeline wordt doorverwezen naar de github van Arthur
https://github.com/Matthino868/TORCS-Self-Driving-Agent

Dit is een fork van CognitiaAI
https://github.com/CognitiaAI/TORCS-Self-Driving-Agent

Er is gekozen om niet alle bestanden in deze github te plaatsen omdat Torcs in zijn geheel op deze github staat. Daarnaast wordt er gebruik gemaakt van een Java client om een auto te besturen. 
Deze github heeft de basis van een data pipeline. Een nadeel is dat er maar 1 server tegelijk start. Hierdoor duurt het genereren van data lang. 

We hebben een aanpassing gemaakt om het genereren van data te versnellen. Alle 10 de SCR-servers kunnen nu tegelijk hun eigen race starten.
Hierdoor kan er 10x zo snel data gegenereerd worden. Alle data wordt ook netjes genummerd en in een mapje met dezelfde naam als de track gestopt.

Om de pipeline te starten moet eerst de data_generation_client.py gestart worden. Dit laat meerdere clients starten op hun eigen portnummer. SCR-client-1 start op 3001, SCR-client-2 start op 3002, enz... Daarna moet de data_generation_server.py gestart worden. Tussen het starten van elke server wordt quickrace.xml aangepast. Server 1 past de starting grid aan zodat SCR-server-1 op een random positie in de race start. Server 2 doet hetzelfde voor SCR-server-2. Enz..  