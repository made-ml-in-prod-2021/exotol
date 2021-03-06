ml_project
==============================

Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
Для тренировки логистической регрессии:
~~~
python ml_project/train_pipeline.py configs/train_config_logistic_regression.yaml
~~~
В конфиге есть пункт 
~~~
feature_params:
    features_and_transformers_map: "configs/features_lr.yaml"
~~~
Важно чтобы по указанному пути лежал файл features_lr.yaml
c указанием каким признакам какие трансформеры нужны

Для обучения random forest нужно выполнить скрипт с
соответствующим конфигом
~~~
python ml_project/train_pipeline.py configs/train_config_randomforest.yaml
~~~
Снова не забываем о 
~~~
feature_params:
    features_and_transformers_map: "configs/features_rf.yaml"
~~~
где должны быть указаны соответствия между трансформерами и признаками


Чтобы выполнить предсказания нужно выполнить 
~~~
python ml_project/predict_pipeline configs/predict_config.yaml
~~~

Test:
~~~
pytest tests/
~~~


В параметре path_to_config
~~~
log_params:
    path_to_config: "configs/logging.yaml"
~~~ 
лежит путь к файлу конфигурации логирования

Есть 5 основных логгеров:
- pipeline
- data
- models
- features
- inference

Чтобы все работало универсально (в PyCharm, в консоле)
Считаем, что все пути должны идти от рабочей директории всего проекта т.е. мы выкачали проект, рабочей директорией будет папка `exotol` == '.'
Все пути будут идти относительно нее.

Каждый используется в соответствующем логическом блоке приложения

-2) ~~Назовите ветку homework1 (1 балл)~~ +1
~~-1) положите код в папку ml_project~~
~~0) В описании к пулл реквесту описаны основные "архитектурные" и тактические решения, которые сделаны в вашей работе. В общем, описание что именно вы сделали и для чего, чтобы вашим ревьюерам было легче понять ваш код. (2 балла)~~ +2

~~1) Выполнение EDA, закоммитьте ноутбук в папку с ноутбуками (2 баллов)~~  +2
~~Вы так же можете построить в ноутбуке прототип(если это вписывается в ваш стиль работы)
Можете использовать не ноутбук, а скрипт, который сгенерит отчет, закоммитьте и скрипт и отчет (за это + 1 балл)~~

~~2) Проект имеет модульную структуру(не все в одном файле =) ) (2 баллов)~~ +2

~~3) использованы логгеры (2 балла)~~ +2

~~4) написаны тесты на отдельные модули и на прогон всего пайплайна(3 баллов)~~ +3

~~5) Для тестов генерируются синтетические данные, приближенные к реальным (3 баллов)~~ +3
~~- можно посмотреть на библиотеки https://faker.readthedocs.io/en/, https://feature-forge.readthedocs.io/en/latest/~~
~~- можно просто руками посоздавать данных, собственноручно написанными функциями~~
~~как альтернатива, можно закоммитить файл с подмножеством трейна(это не оценивается)~~ 

~~6) Обучение модели конфигурируется с помощью конфигов в json или yaml,  
   закоммитьте как минимум 2 корректные конфигурации, с помощью которых можно 
   обучить модель (разные модели, стратегии split, preprocessing) (3 балла)~~ +3

~~7) Используются датаклассы для сущностей из конфига, а не голые dict (3 балла)~~ +3

~~8) Используйте кастомный трансформер(написанный своими руками) и протестируйте его(3 балла)~~ +3

~~9) Обучите модель, запишите в readme как это предлагается (3 балла)~~ +3

~~10) напишите функцию predict, которая примет на вход артефакт/ы от обучения,  
    тестовую выборку(без меток) и запишет предикт, напишите в readme как это сделать (3 балла)~~  +3 

11) Используется hydra  (https://hydra.cc/docs/intro/) (3 балла - доп баллы)

12) Настроен CI(прогон тестов, линтера) на основе github actions 
    (3 балла - доп баллы (будем проходить дальше в курсе, но если есть 
    желание поразбираться - welcome)
~~13) Проведите самооценку, опишите, в какое кол-во баллов по вашему мнению стоит
    оценить вашу работу и почему (1 балл доп баллы)~~

Если посчитал правильно, то получилось 31 балл

Гидру пробовал, но реализовывать не стал, хотя в целом он хорошо ложится на подход с сущностями.
CI тоже настраивать не стал


A short description of the project.

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`│
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
