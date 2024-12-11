# Trainen van Neuraal Netwerk met PyTorch

Dit script is ontworpen om een neuraal netwerk te trainen op basis van gegevens uit CSV-bestanden met behulp van PyTorch.

## Gebruik

Om het script uit te voeren, kun je de volgende command gebruiken:

```bash
python build_pipeline.py <modelnaam> [-p MODEL_PATH] [-d DATA_PATH] [-s SAVE_EPOCH] [-e EPOCHS] [-l FILE_LIMIT] [--graph | --no-graph]
```


### Argumenten

- `<modelnaam>`: Naam van het model dat getraind moet worden.
- `-p, --model-path MODEL_PATH`: Aangepaste bestemmingsmap om het model op te slaan (standaard: "..\models").
- `-d, --data-path DATA_PATH`: Aangepaste map waar de trainingsgegevens vandaan komen (standaard: "..\csv\").
- `-s, --save-epoch SAVE_EPOCH`: Aantal epochs tussen elk modelopslag (standaard: 10).
- `-e, --epochs EPOCHS`: Totaal aantal epochs om te trainen (standaard: 100).
- `-l, --file-limit FILE_LIMIT`: Maximaal aantal oude bestanden voordat ze worden verwijderd (standaard: 5).
- `--graph / --no-graph`: Optie om grafieken weer te geven tijdens het trainen (standaard: --graph).
- `--help`: Toont hulp voor argumenten in het script.

## Voorbeeld 
```bash
python build_pipeline.py my_model -s 15 -e 50 -l 3 --no-graph
```
Dit voorbeeld laat zien hoe je het script uitvoert met aangepaste opties.

### Uitleg voorbeeld
- my_model: Dit is de naam van het model dat getraind wordt.
- -s 15: Het model wordt opgeslagen na elke 15 epochs.
- -e 50: Het totale aantal epochs om te trainen is 50.
- -l 3: Er worden maximaal 3 oude modellen opgeslagen voordat de oudste worden verwijderd.
- --no-graph: Grafieken worden niet weergegeven tijdens het trainen.
