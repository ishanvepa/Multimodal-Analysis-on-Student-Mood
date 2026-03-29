### Google Review Images

Download images from [Google drive](https://drive.google.com/file/d/1rrGMIap6TmVyX9I59e9uPLgAMUw_2hhC/view?usp=sharing).  
Filename: G_{school}\_{index of the google review file}\_{index of the review in that file}#{index of the photo in that review}.ext = {Unique_ID}#{index of the photo in the review}.ext


### Place perception
```bash
pip install zensvi 1.4.7
```

in line 275 of `{your site-packages path}/zensvi/cv/classification/perception.py` add this line, otherwise, models will not be successfully loaded

```python
checkpoint_path = Path(__file__).parent.parent.parent.parent.parent / model_load_path / file_name
```
then run
```python
python place_perception.py
```
