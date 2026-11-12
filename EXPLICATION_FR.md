# Explication simple du projet « Décrochage d'étudiants avec un LSTM »

## 1. L'idée en une phrase

Dans un cours en ligne, on regarde combien chaque étudiant clique sur la plateforme
**semaine après semaine**, et on essaie de prédire **le plus tôt possible** s'il va
abandonner ou échouer, pour qu'un enseignant puisse l'aider à temps.

## 2. Les données : OULAD

- **OULAD** = *Open University Learning Analytics Dataset*. L'Open University est une grande
  université à distance au Royaume-Uni. Elle a publié des données anonymisées de 2013–2014.
- **Licence CC BY 4.0** : on peut réutiliser les données à condition de citer les auteurs
  (Kuzilek et al., 2017). On ne les met pas sur GitHub : le script les télécharge.
- **32 593 inscriptions** (un étudiant inscrit à un module = une inscription),
  7 modules, 22 sessions de cours.
- **10 655 280 lignes de clics** : « l'étudiant X a cliqué 4 fois sur la ressource Y le jour 12 ».
- **VLE** (*Virtual Learning Environment*) = la plateforme du cours (comme Moodle).

## 3. La cible (ce qu'on prédit) et pourquoi

Chaque étudiant a un résultat final : *Distinction* (mention), *Pass* (réussi), *Fail* (échoué),
*Withdrawn* (a abandonné). On en fait **deux classes** :

- **1 = « à risque »** : Withdrawn ou Fail ;
- **0 = « réussite »** : Pass ou Distinction.

Pourquoi ? Pour un enseignant, abandonner ou échouer veut dire la même chose : l'étudiant
aurait eu besoin d'aide. Et sur le graphique des clics, ces deux groupes se comportent
pareil dès les premières semaines. En plus, on obtient deux classes presque équilibrées
(47,8 % / 52,2 %).

**Filtrage** : 3 097 étudiants se sont désinscrits **avant le début du cours**. Ils n'ont jamais
pu cliquer, les prédire n'a aucun intérêt → on les enlève. Il reste **29 496** inscriptions.

## 4. Construire une séquence par étudiant

On découpe le temps en **semaines** (jour 0 à 6 = semaine 1, etc.) et on garde les
**15 premières semaines** (≈ 40 % d'un cours de 9 mois). Pour chaque semaine, 2 nombres :

- le nombre total de clics (on prend le **logarithme**, `log(1 + clics)`, pour que 1 000 clics
  n'écrasent pas tout : log(1+1000) ≈ 6,9 alors que log(1+10) ≈ 2,4) ;
- le nombre de jours actifs (0 à 7).

Exemple pour un étudiant :
`semaine 1 : 40 clics, 4 jours` → `semaine 2 : 12 clics, 2 jours` → `semaine 3 : 0 clic, 0 jour`…
C'est une **séquence** : l'ordre compte.

## 5. Séparer les données (train / validation / test)

- **Train (70 %)** : les exemples pour apprendre.
- **Validation (10 %)** : pour décider quand arrêter l'entraînement du LSTM.
- **Test (20 %, 5 913 inscriptions)** : des étudiants jamais vus, pour la note finale.
- **Groupé par étudiant** : si un même étudiant suit deux modules, ses deux inscriptions vont du
  même côté. Sinon le modèle « reconnaîtrait » l'étudiant → résultat trop optimiste.

## 6. Les 3 modèles comparés

1. **LR total** (*régression logistique sur les clics totaux*, la base demandée) :
   un seul nombre par étudiant, le total des clics. Exemple : 300 clics en 10 semaines.
   La régression logistique transforme ce nombre en probabilité d'être à risque.
2. **LR weekly** : régression logistique sur les valeurs **de chaque semaine** mises côte à côte
   (sem1, sem2, …). Elle a la même information que le LSTM mais ne « comprend » pas l'ordre.
3. **LSTM** (*Long Short-Term Memory*) : un réseau de neurones **récurrent**. Il lit les semaines
   **une par une** et garde une petite **mémoire** (32 nombres) qu'il met à jour à chaque semaine.
   À la fin, la mémoire est transformée en une probabilité.
   Image simple : un professeur qui lit le carnet de présence semaine par semaine et se dit
   « il venait beaucoup au début, puis de moins en moins… attention ».

Mots techniques de l'entraînement :
- **Époque** (*epoch*) : un passage complet sur toutes les données d'entraînement.
- **Batch** : petit paquet de 256 étudiants traités ensemble.
- **Adam** : la méthode qui ajuste les poids du réseau pour réduire l'erreur.
- **Perte BCE** : mesure de l'erreur pour un problème oui/non.
- **Dropout 0,2** : on « éteint » au hasard 20 % des neurones pendant l'entraînement pour éviter
  d'apprendre par cœur.
- **Early stopping** : on arrête quand le score sur la validation ne s'améliore plus depuis
  8 époques, et on garde la meilleure version.
- **PyTorch sur CPU** : tout tourne sur le processeur, sans carte graphique (≈ 7 minutes au total).

## 7. Les métriques

- **Accuracy** : % de bonnes réponses. Exemple 0,787 = 78,7 % des étudiants bien classés.
- **F1** : équilibre entre « on trouve beaucoup d'étudiants à risque » (rappel) et
  « quand on dit à risque, c'est vrai » (précision). Calculé avec un seuil de 0,5.
- **ROC-AUC** : si on prend au hasard un étudiant à risque et un étudiant qui réussit, c'est la
  probabilité que le modèle donne un score plus élevé au premier. 0,5 = hasard ; 1 = parfait.

## 8. Les résultats (vrais chiffres)

| Semaines observées | LR total | LR weekly | LSTM |
|---|---:|---:|---:|
| 5 (ROC-AUC) | 0,689 | 0,723 | **0,732** |
| 10 (ROC-AUC) | 0,726 | 0,800 | **0,808** |
| 15 (ROC-AUC) | 0,754 | 0,854 | **0,859** |

À 15 semaines, le LSTM a une accuracy de 0,787 et un F1 de 0,748 (LR weekly : F1 0,762).

**La courbe « performance selon la semaine »** (figure 03) montre que :
- dès la **semaine 1**, on fait mieux que le hasard (AUC ≈ 0,65) ;
- le LSTM dépasse **0,75 à la semaine 6** et **0,80 à la semaine 10** ;
- plus on attend, mieux on prédit… mais plus il est tard pour aider l'étudiant. C'est le
  **compromis** à expliquer en entretien.

**Version honnête (lignes en pointillés)** : à la semaine *w*, certains étudiants ont déjà quitté
le cours, et ils sont faciles à repérer (0 clic). Si on ne garde que ceux **encore inscrits**,
l'AUC du LSTM est 0,699 (semaine 5), 0,755 (semaine 10), 0,803 (semaine 15).

## 9. Ce qu'il faut retenir (message pour l'entretien)

1. **Le moment des clics compte plus que leur quantité** : passer du total des clics aux clics
   par semaine fait gagner +0,074 d'AUC à la semaine 10.
2. **Le LSTM n'apporte qu'un petit gain** (+0,004 à +0,009 d'AUC) par rapport à une régression
   logistique sur les semaines, et son F1 est un peu plus bas avec le seuil 0,5. Avec seulement
   2 informations par semaine, un modèle simple suffit presque. Un LSTM serait plus utile avec
   plus d'informations (type de ressource, notes des devoirs).
3. Je ne cache pas ce résultat : savoir dire « le modèle complexe n'apporte pas beaucoup ici »
   est une qualité de chercheur.

## 10. Limites

- Seulement les clics (pas les notes ni le profil de l'étudiant).
- Seuil 0,5 non optimisé.
- Découpage aléatoire et non temporel (idéal : entraîner sur 2013, tester sur 2014).
- Une seule graine aléatoire : une différence de moins de 0,01 n'est pas forcément significative.
- Une seule université britannique.

## Petit lexique

- **Learning analytics** : analyse des traces numériques des apprenants pour améliorer
  l'apprentissage.
- **Décrochage** (*dropout*) : un étudiant qui abandonne. Attention : en deep learning,
  *dropout* veut aussi dire « éteindre des neurones au hasard » — deux sens différents !
- **Séquence / série temporelle** : des valeurs rangées dans l'ordre du temps.
- **Réseau récurrent (RNN)** : réseau qui traite une séquence pas à pas avec une mémoire.
  Le LSTM est un RNN amélioré qui oublie moins vite.
