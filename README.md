# Travail de Fin d'Etude
Ici, nous parlons du code de mon mémoire qui a pour but prémière d'abord d'implémenter une solution YOLOv8 et DeepSORT 
L'algorithme permet de détecter et suivre en temps réel les actions violentes et suspects des personnes dans un enviromment comme la faculté polytechnique. 

Dans le repertoire Try_3_Train. Nous avons pu effectivement entrainer notre modèle avec de données qui ont comme classes 'NonViolence', 'Violence', 'guns', 'knife'

Dans le repertoire Try_2_OK. Où se trouve notre inférence du programme qui tourne avec succès avec différents possibilités à considérer. 

Les dépendances utiles pour l'exécution de l'implementation :

ultralytics == 8.2.2 ;
deep-sort-realtime ==	1.3.2;
torch == 2.2.2 ;
torchaudio == 2.2.2 ;
torchvision == 0.17.2 ;
nvidia-cublas-cu12 ==	12.5.3.2; 
nvidia-cuda-nvrtc-cu12	== 12.5.82; 
nvidia-cudnn-cu12 ==	9.2.1.18; 

Ces différents packages sont très utiles pour le bon fonctionnement de l'algorithme. 
