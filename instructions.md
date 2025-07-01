Hey, Felix!

Here are the istructions for running this project (at least until now) and is by running all commands in your terminal:

0. Before everything, add to the big folder the data I send you on private. There are 2 files: "waves123_augmented_consent.dta" and "data_normalized_fr". AND generate an empty folder also in the same space called: bld. You can do this also like this:
```bash
mkdir bld
```

1. Be sure you have a system that allows you to develop and choose environments, in my case, I use MAMBA (highly recommend).

2. Once you have this, you first have to generate the environment, for this you should type this command:

```bash
conda env create -f env_ml.yml
```
(This step should take some minutes...)

3. After this is done, you should activate your environment typing:

```bash
conda activate ml_project
```
(The environment should appear on the left of your cd now)

4. The next step is to run the files needed to create the images or results, right now these are only images:
```bash
python r_code_in_python_delete.py
python whole_project.py
```

5. You can also go to the notebooks, for example "vamos_de_a_poco.ipynb" and if you select the environment created ("ml_project") you can run it little by little. You just need to make sure you have all the files I am sending in the same big folder.

Please let me know if any mistake or if something is unclear :)