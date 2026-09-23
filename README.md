# EntityFuture

Eine [HACS](https://hacs.xyz/)-Integration für [Home Assistant](https://www.home-assistant.io/), die den Zustand einer Entität **X Minuten in der Zukunft** vorhersagt – mit einem bewusst einfachen, selbstlernenden Modell (Naive Bayes), das direkt in Home Assistant läuft, ohne externe Abhängigkeiten (kein NumPy, kein scikit-learn, keine Cloud).

## Was macht EntityFuture?

Du legst einen "Prädiktor" an:

- **Ziel-Entität** – z. B. `binary_sensor.badezimmer_feucht`, `person.sven`, `switch.heizung`
- **Zielzustand** – welcher Zustand als "Ja" zählt, z. B. `on` oder `home`
- **Horizont** – wie viele Minuten in die Zukunft vorhergesagt werden soll, z. B. 30

Optional fügst du **Hilfs-Entitäten** hinzu, die beim Lernen helfen sollen (z. B. `binary_sensor.schulferien`, `person.partner`, `sensor.wetter_zustand`, `input_boolean.homeoffice`).

Daraus entsteht ein Gerät mit drei Entitäten:

| Entity | Beschreibung |
|---|---|
| `sensor.<name>_wahrscheinlichkeit` | Wahrscheinlichkeit (0–100 %), dass die Ziel-Entität in *X* Minuten den Zielzustand hat |
| `binary_sensor.<name>_vorhersage` | Ja/Nein, basierend auf einem konfigurierbaren Schwellenwert (Standard 50 %) |
| `sensor.<name>_genauigkeit` | Rollierende Trefferquote der letzten Vorhersagen (Diagnose-Entity) |

## Wie funktioniert das Modell?

**Naive Bayes**, von Hand implementiert, ganz ohne ML-Bibliothek:

- Als Merkmale ("Features") fließen ein: Wochentag, Tageszeit (30-Minuten-Raster), der aktuelle Zustand der Ziel-Entität selbst sowie der aktuelle Zustand jeder Hilfs-Entität.
- Alle paar Minuten (Abtastintervall, Standard 5 Min.) nimmt die Integration einen Snapshot dieser Merkmale auf und merkt sich: "schau in *X* Minuten nach, was aus der Ziel-Entität geworden ist".
- Sobald dieser Zeitpunkt erreicht ist, wird der tatsächliche Zustand mit der Vorhersage verglichen (das ergibt die **Genauigkeit**, ganz ehrlich, ohne Testdaten-Trickserei – "predict then verify") und der Merkmalsvektor zusammen mit dem beobachteten Ergebnis ins Modell eingelernt.
- Es werden nur Zähler pro Merkmalswert und Klasse geführt – Speicherbedarf wächst nur mit der Anzahl unterschiedlicher beobachteter Zustände, nicht mit der Zeit. Ressourcenschonend genug für einen Raspberry Pi.
- Beim Einrichten kann optional vorhandene [Recorder](https://www.home-assistant.io/integrations/recorder/)-Historie (Standard: letzte 7 Tage) genutzt werden, um das Modell "warmzustarten", statt bei null anzufangen.

Das Modell und alle offenen (noch nicht verifizierten) Vorhersagen werden persistiert und überleben einen Neustart von Home Assistant.

## Installation

### Über HACS (benutzerdefiniertes Repository)

1. HACS → Integrationen → Menü (⋮) → *Benutzerdefinierte Repositories*
2. Repository-URL: `https://github.com/SrSupp/EntityFuture`, Kategorie: *Integration*
3. "EntityFuture" installieren, Home Assistant neu starten

### Manuell

`custom_components/entityfuture` in dein `config/custom_components/`-Verzeichnis kopieren und Home Assistant neu starten.

## Einrichtung

1. Einstellungen → Geräte & Dienste → Integration hinzufügen → *EntityFuture*
2. Name, Ziel-Entität und Horizont (Minuten) angeben
3. Zielzustand bestätigen/anpassen (vorbelegt mit dem aktuellen Zustand der Entität)
4. Danach über *Konfigurieren* am Gerät jederzeit Hilfs-Entitäten, Abtastintervall, Schwellenwert und den Recorder-Warmstart anpassen

## Dienst `entityfuture.reset_model`

Setzt das gelernte Modell eines Prädiktors zurück (z. B. nach dem Hinzufügen neuer Hilfs-Entitäten, wenn du "von vorn" lernen willst) und kann optional den Recorder-Warmstart erneut ausführen.

## Grenzen (bewusst, für v1)

- Nur diskrete Zustände als Ziel (kein "Temperatur > 22 °C"-Schwellenwert auf numerischen Sensoren)
- Ein Prädiktor = eine Ziel-Entität + ein Horizont; für mehrere Horizonte einfach mehrere Prädiktoren anlegen
- Naive Bayes nimmt an, dass Merkmale bedingt unabhängig sind – für eng korrelierte Hilfs-Entitäten kann das die Wahrscheinlichkeit leicht verzerren, in der Praxis für dieses Szenario aber ein guter, sehr günstiger Kompromiss

## Entwicklung

```bash
pip install pytest
pytest tests/
```

`tests/test_model.py` prüft die Kernlogik des Naive-Bayes-Modells unabhängig von Home Assistant. Der Rest der Integration (Config-Flow, Coordinator, Recorder-Import) sollte zusätzlich in einer echten Home-Assistant-Instanz getestet werden.
