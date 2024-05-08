# Science

Classifies images that are collected from a bottom camera on a Sage/Waggle node. The method is based on the MobileNet model. It was created as an example how [KERAS](https://keras.io/) can be used with Sage to easily create and test Models for the Edge.

# AI at Edge

The code runs a MobileNet based model with a given time interval. In each run, it takes a still image from a given camera (bottom) and outputs if flooding is detected or not (1 or 0). The plugin resizes images to 224x224 as the model was trained with the size.

# Ontology

The code publishes measurements with a topic of `env.binary.flood`.

# Inference from Sage codes
To query the output from the plugin, you can do so with the python library 'sage_data_client':
```
import sage_data_client

# query and load data into pandas data frame
df = sage_data_client.query(
    ## start and end date option 1
    start="yyyy-mm-dd",
    end="yyyy-mm-dd",
    ## start and end date option 2
    start="-1h",

    filter={
        "name": "env.binary.flood",
    }
)

# print results in data frame
print(df)
# print results by its name
print(df.name.value_counts())
# print filter names
print(df.name.unique())
```
For more information, please see [Access and use data documentation](https://docs.sagecontinuum.org/docs/tutorials/accessing-data) and [sage_data_client](https://pypi.org/project/sage-data-client/).