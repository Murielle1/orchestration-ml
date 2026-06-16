# ✈️ Airline Passenger Satisfaction — MLOps Pipeline

> Projet de classification binaire avec orchestration ML et déploiement cloud

---

## 📌 Contexte & Problématique

Dans un secteur aérien hautement concurrentiel, la satisfaction client est un levier stratégique majeur. Les compagnies aériennes collectent de grandes quantités de données à chaque vol — profil du passager, classe de voyage, retards, qualité des services à bord — mais peinent souvent à exploiter ces informations de manière prédictive et actionnable.

**Comment, à partir des données d'un vol et du profil d'un passager, peut-on prédire automatiquement son niveau de satisfaction afin d'anticiper les insatisfactions et améliorer l'expérience client ?**

Ce projet répond à cette question en construisant un pipeline MLOps complet : de l'entraînement d'un modèle de classification binaire jusqu'à son déploiement et son exposition via une API sur le cloud.

---

## 🎯 Objectif

Entraîner, évaluer et déployer un modèle de **classification binaire** capable de prédire si un passager est :

- `satisfied` — passager satisfait
- `neutral or dissatisfied` — passager neutre ou insatisfait

Le modèle doit être reproductible, monitorable et exploitable en production via une API REST.

---

## 📊 Données

**Source :** [Airline Passenger Satisfaction — Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

Le jeu de données contient plus de **100 000 observations** réparties en un jeu d'entraînement (80 %) et un jeu de test (20 %), avec **22 variables prédictives** de nature mixte (numériques et catégorielles).
