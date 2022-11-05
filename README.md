# Prototyp-Implementierung für meine Bachelorarbeit 

Code meiner Bachelorarbeit mit dem Titel "Implementierung und Validierung von computergenerierten
abstrakten Portraits mithilfe von CycleGANs" im Studiengang Media Engineering.

Dieses Repository enthält die Implementierung von computergenerierten abstrakten Portraits, die mit der Deep-Learning-Methode CycleGAN entwickelt wurde. 

Die restlichen Ordner, in denen die generierten Bilder gespeichert sind, sind nicht in diesem Repository, da die Anzahl der Bilder sehr hoch ist. 
Z. B. die generierten abstrakten Ergebnisse oder die Datensätze. 

## Beschreibung 

Dieser Prototyp generiert Menschenportraits in einem abstrakten Stil. Dabei werden 2 Datensätze, abstrakte Bilder und Menschenportraits, benutzt.
Der Prototyp soll nachweisen, ob sich die Methode CycleGAN für die Umwandlung von Menschenportraits in abstrakte Portraitkunstwerke eignet.
Die Implementierung des Prototypen ist in Python geschrieben und benutzt das Framework Pytorch.

## Installation 

1. Sicherstellen, dass Python, PyTorch und Cuda installiert ist. 
2. Repo klonen  
3. Einen Ordner namens "saved_images" erstellen in der gleichen Hierarchie, wie alle Dateien.
4. Code ausführen mit python3 Train.py --dataroot datasets/ --cuda


## Github-Repository Quellen 

https://github.com/junyanz/CycleGAN

https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/CycleGAN

https://github.com/aitorzip/PyTorch-CycleGAN


~ Susanna Lankocz
