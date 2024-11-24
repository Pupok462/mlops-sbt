# ДЗ-2: Hydra, DVC, Lightning

Используя результаты ДЗ-1, имплементируйте цикл обучения на pytorch lightning, используйте hydra как точку входа, подключите DVC для хранения файлов

**Definition of done:**
- Step-1 **[1 балл]**: Используя python модуль с биндингами напишите необходимый код для лайнтнинга ```pl.LightningDataModule, pl.LightningModule```. Биндинги должны использоваться в подготовке данных (```__getitem__ ```метод класса ```torch.utils.data.Dataset```).
- Step-2 **[3 балла]**: точка входа - скрипт ```train.py```, под main guard - вызов единственного метода с декоратором hydra. Конфиги hydra - в отдельной директории, разбитые на логические файлы (в один ```.yaml``` нельзя). В коде нет никаких литератов/констант -- все значения должны быть получены из ```yaml``` с конфигом
- Step-3 **[3 балла]**: подключен dvc (с любым из типов remote storage), в него загружено несколько файлов. Будет проверяться наличие и валидность ```/.dvc``` и ```.dvc``` файлов.

Итоговая оценка за задание - сумма баллов от первого пункта до последнего выполненного без пропусков (то есть если например выполнен только пункт 3 то сумма баллов - ноль).

**Дедлайн: 30 ноября 23:59**

Форма сдачи - ссылка на репозиторий.
Случайные 30% студентов будут сдавать онлайн. Об этом будет сообщено ближе к концу курса, чтобы провести сдачу всех ДЗ на одной встрече.