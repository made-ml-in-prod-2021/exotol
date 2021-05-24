ml_project
==============================

Проект `heart_disease_common`
содержит общие артифакты для других проектов:
- heart_disease_train
- online_inference


Установка:

~~~
python -m venv .venv
source .venv/bin/activate # на Linux
.venv/Scripts/activate # на Windows
pip install .
~~~

Чтобы запустить тестирование, нужно установить
в окружение дополнительные библиотеки
~~~
pip install .[tests]
~~~

Папка `scripts` содержит cmd скрипт для запуска 
тестов (считаем, что вначале была выполнена установка 
соответствующих библиотек)

