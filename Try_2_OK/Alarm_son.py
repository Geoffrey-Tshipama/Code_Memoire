from pynput import keyboard
import pygame

# Initialiser pygame mixer pour jouer des sons
pygame.mixer.init()

# Charger le son
pygame.mixer.music.load('son.wav')  # Remplacez 'sound.mp3' par le chemin du fichier audio souhaité

def on_press(key):
        try:
            # Arrêter le son si la touche 'q' est pressée
            if key.char == 'q':
                pygame.mixer.music.stop()
            else:
                # Jouer le son à chaque autre appui de touche
                pygame.mixer.music.play()
        except AttributeError:
            # Ignore les touches spéciales (comme shift, ctrl, etc.)
            pass

# Créer un listener pour le clavier
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
