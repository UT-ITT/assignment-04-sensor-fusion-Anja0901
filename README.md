[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/AktWbCri)
# assignment-04-CV-Sensor-Fusion

image_extractor.py 

Dieses Programm lädt ein Bild mit OpenCV, zeigt es an und erlaubt dem Nutzer,
vier Eckpunkte per Mausklick auszuwählen. Aus dem markierten Bereich wird ein
perspektivisch verzerrtes Bild erzeugt und angezeigt.

Bedienung:
Der Benutzer wählt die vier Eckpunkte in folgender Reihenfolge:

1. oben links
2. oben rechts
3. unten rechts
4. unten links

- Linksklick: Eckpunkt auswählen
- ESC: Auswahl verwerfen und neu beginnen
- S: Ergebnis in der Ausgabedatei speichern


Aufrufbeispiel:
python perspective_transformation/image_extractor.py --input perspective_transformation/sample_image.jpg --output perspective_transformation/result.jpg --width 800 --height 600

---------------------------------------------------------------------

AR_game.py

Dieses Spiel erkennt das Marker-Board (das wir bekommen haben) und transformiert das Webcam Bild so, dass ein virtuelles Spielfeld innerhalb der Marker entsteht. Auf diesem Spielfeld erscheint ein zufälliger Punkt, der möglichst schnell berührt werden soll.
Je nach Beleuchtung und Situation kann die Erkennung der Berührung entweder erfolgen durch:
- den Finger
- einem dunklen Objekt z.b. Stift

Die erkannte Kontur wird dem Spieler angezeigt.

Wurde der Punkt erwischt, erscheint ein neuer Punkt an einer neuen zufälligen Position. Dies geschieht 20 Mal. Dabei wird die Zeit gemessen. Wurden alle 20 Punkte getroffen, endet das Spiel und ein Screen mit der dafür benötigten Zeit erscheint.
Ziel des Spieles ist es diese Zeit möglichst gering zu halten.

Tasten:
- R: Neustart
- M: Maske wechseln

Masken:
- Helligkeitsmaske: Die Maske erkennt die Kontur anhand dunkler Bereiche im Bild
- Hautfarbenmaske: Die Maske erkennt die Kontur anhand Hautfarbener Bereiche im Bild

Standardmäßig ist die Hautfarbenmaske aktiv. Diese funktioniert besonders gut für die Fingererkennung, bei Tageslicht oder weißem Licht.
Die Helligkeitsmaske funktioniert besonders gut für dunkle Gegenstände und ist robuster, in Bezug auf die konkrete Lichtfarbe.

---------------------------------------------------------------------

sensor_fusion.py

Zuerst muss das Marker Board in die Kamera gehalten werden. Dann werden die Marker erkannt und das Bild entsprechend extrahiert. Wird dann das Handy mit dem Marker und Dippid in das Bild gehalten, wird ein Roter Punkt auf dem Marker angezeigt. Dieser ist zeigt die absolute Position des Handys welche über den Marker erkannt wird. Zusätzlich erscheint ein grüner Punkt. Dieser ergibt sich durch eine Kombination aus einer kamerabasierten Markererkennung mit Beschleunigungsdaten durch einen komplementären Filter. Der Parameter alpha bestimmt dabei, wie stark die Kameraposition beziehungsweise die aus den Beschleunigungsdaten vorhergesagte Position gewichtet wird. Über die Pfleitasten (Up,Down) kann der alpha Wert angepasst werden. Über den Button 1 von Dippid lässt sich der grüne Punkt auf die Markerposition setzen.

Hohe alpha Werte führen dazu, dass die Vorhersage der Kameraposition stärker folgt und dadurch stabiler ist. Niedrigere α-Werte geben den Beschleunigungsdaten mehr Einfluss. Aufgrund von Messrauschen und Drift führt das allerdings auch zu ungenaueren Vorhersagen. In den Tests lieferten höhere α-Werte insgesamt die zuverlässigsten Ergebnisse.