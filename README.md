# crowd_density_estimation
model for crowd_density_estimation

## Installation
Install the packages required for development.
```bash
conda create -n density python=3.10
conda activate density
git clone https://github.com/Hzzz123rfefd/crowd_density_estimation.git
cd crowd_density_estimation
pip install -r requirements.txt
```


## Usage
### Dataset
Firstly, you can download the ShanghaiTech dataset  
[ShanghaiTech](https://paperswithcode.com/dataset/shanghaitech)
then, you can put the `ShanghaiTech` folder into `img` folder
your directory structure should be:
- crowd_density_estimation/
  - imgs/
    - ShanghaiTech/

### Trainning
An examplary training script is provided in `train.py`.
You can adjust the model parameters in `config/density.yml`
```bash
python train.py --model_config_path config/density.yml
```

### Inference
Once you have trained your model, you can use the following script to generate crowd density map:
```bash
python example/inference.py --image_path {yout image path} --model_config_path config/density.yml --output_path {density map path}
```

## A real-time crowd density monitoring system
### start your model service
Once you have trained your model, you can deploy your model as a web service:
```bash
python example/deploy.py  --model_config_path config/density.yml --host "127.0.0.1" --port "8000"
```
### Start your monitoring app
After deploying your model web service, you can run your app:
```bash
streamlit run app.py
```

## Related links
 * ShanghaiTech Dataset: https://paperswithcode.com/dataset/shanghaitech

