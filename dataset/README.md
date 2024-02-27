# CORE Datasets 
### Structure
The datasets **CQPy**, **CQPyUs** and the **SQJava** are as mentioned in our [research](https://aka.ms/COREMSRI), a subset of the [CodeQueries](https://huggingface.co/datasets/thepurpleowl/codequeries) dataset (CQPy, CQPyUs) and subset of the [Sorald Dataset](https://github.com/khaes-kth/Sorald-experiments). The overall structure of these are as follows.

```
<dataset>
    <query>
        <1.py/1.java>
        <2.py/2.java>
        .
        .
        .
        <n.py/n.java>
        result.csv
```

Where `<dataset>` refers to dataset being used in question, `<query>` is the folder corresponding to the files with warnings from corresponding query. Each query subfolder contains `n` number of files which constitute the parent files that have flagged checks alongwith a `results.csv` file which summarizes the checks flagged in all the files within the query subfolder in format similar to CodeQL output.

To extract these datasets, simply run from the project directory

```bash
tar -xvf dataset/<dataset>.tar.gz -C dataset
```

to extract the dataset in the `dataset` dir. The structure defined above is how our scripts expect the files to be present and thus simply run our various scripts on these datasets by giving them as arguments as `dataset/<dataset>` (from project dir).