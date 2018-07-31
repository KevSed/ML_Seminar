# Seminar Maschinelles Lernen für Physikstudierende
Machine learning seminar at TU Dortmund, summer term 2018.

Das Repository enthält 6 verschiedene **network{}.py** files. Diese ausgeführt trainieren
verschiedene Netzwerkarchitekturen auf Bildern, die in HDF5 files gespeichert sind.
Diese files werden sehr schnell sehr groß und benötigen daher auch sehr viel Arbeitsspeicher.
Der Aufwand nimmt für komplexere Netzstrukturen auch weiter zu.
Mit Hilfe des **load_data.py** Skriptes lassen sich Daten aus einer Ordnerstruktur gemäß der
verschiedenen Klassen einlesen und in HDF5 files speichern.
Zunächst muss dafür der Datensatz heruntergeladen und entpackt werden.
Der Download kann über die website [kaggle](https://www.kaggle.com/paultimothymooney/kermany2018)
oder direkt von der [Quelle](https://data.mendeley.com/datasets/rscbjbr9sj/2/files/9e8f7acf-7d3a-487f-8eb5-0bd3255b9685/OCT2017.tar.gz?dl=1) erfolgen.
```
$ wget https://data.mendeley.com/datasets/rscbjbr9sj/2/files/9e8f7acf-7d3a-487f-8eb5-0bd3255b9685/OCT2017.tar.gz?dl=1
```
Um daraus ein HDF5 file zu generieren lautet die Syntax:
```
$ python load_data.py input_file output_file
```
Das Parallelisieren ist hier leider nicht lauffähig.
Die so generierten Daten können den network Skripten zur Verfügung gestellt werden. Dazu muss der entsprechende Pfad im Skript gesetzt werden.
Nach dem Trainieren werden die Modelle, sowie die verwendeten Gewichte und der Validierungsdatensatz gespeichert. Die dazu verwende Funktion ist in dem file **save_all.py** definiert.

Um performance plots und andere Statiskien zu generieren, existieren mehrere Funktionen. Diese sind in den files **evaluate.py** und **plot_history.py** definiert und weiter beschrieben. Sie werden durch **plot_statistics.py** ausgeführt.

Die **Alternativmethode** ist im Ordner Alternative implementiert. **[BESCHREIBUNG]**
