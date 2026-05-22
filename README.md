# Decision Support Autonomous Vehicles

Guide de lancement de l'entrainement CARLA RL en mode online, avec une execution qui continue meme si VS Code est ferme.

## Pre-requis

- Linux sur la VM
- CARLA lance sur le port `2000`
- Le virtualenv `carla_env_37` disponible a la racine du projet

## 1. Demarrer CARLA

Depuis le dossier CARLA, lancer le serveur avec le port `2000`.

Exemple:

```bash
./CarlaUE4.sh -quality-level=Low -world-port=2000
```

## 2. Preparer les environnements Python

CARLA reste dans l'environnement Python 3.7:

```bash
carla_env_37/bin/python -c "import carla; print('CARLA API OK')"
```

L'entrainement RL se lance depuis l'environnement Python 3.10 du projet. Remplace `<py310_venv>` par le chemin de ton venv Python 3.10.

## 3. Verifier la connexion a CARLA

Avant de lancer l'entrainement, verifier que le simulateur repond:

```bash
carla_env_37/bin/python test_carla.py
```

Si tout va bien, le script doit afficher une connexion reussie, la carte chargee et le nombre d'acteurs.

## 4. Lancer l'entrainement de maniere detachee

Pour que l'entrainement continue apres la fermeture de VS Code, il faut le lancer en arriere-plan avec `nohup`.

```bash
mkdir -p outputs/logs
nohup <py310_venv>/bin/python -u -m src.rl.online.train_online_carla_remote > outputs/logs/train_online_carla.log 2>&1 &
echo $!
```

Le `echo $!` affiche le PID du process. Tu peux fermer VS Code apres cette commande: l'entrainement continue dans la VM tant que celle-ci reste allumee.

Dans ce mode, le process Python 3.10 fait l'entrainement RL, le replay buffer, l'optimisation DQN, le reward et la sauvegarde des checkpoints. Il lance seulement un worker `carla_env_37/bin/python` pour piloter CARLA et recuperer les observations.

## 5. Suivre l'avancement

Pour voir les logs en direct:

```bash
tail -f outputs/logs/train_online_carla.log
```

Le script ecrit aussi les resultats dans:

- `outputs/carla/online/<scenario>/online_training_results.csv`

Les modeles sauvegardes sont ecrits dans:

- `models/online/<scenario>_dqn_model.pth`

## 6. Arreter proprement l'entrainement

Si tu veux stopper le run, utilise le PID affiche par `echo $!`:

```bash
kill <PID>
```

Si besoin, tu peux forcer l'arret avec:

```bash
kill -9 <PID>
```

## Option alternative: tmux

Si `tmux` est installe sur la VM, tu peux aussi l'utiliser pour garder une session interactive persistante:

```bash
tmux new -s carla_train
<py310_venv>/bin/python -u -m src.rl.online.train_online_carla_remote
```

Pour detacher la session sans arreter le training: `Ctrl+b`, puis `d`.

Pour revenir dessus:

```bash
tmux attach -t carla_train
```

## Remarque

Le training actuel est un entrainement RL online avec CARLA, sans dataset predefini. Le VLM n'est pas necessaire pour ce point d'entree.
