Apache Spark installation + ipython notebook integration guide for Mac OS X
===========================================================================

Tested with Apache Spark 1.3.1, Python 2.7.9 and Java 1.8.0_45


Install Java Development Kit
----------------------------
Download and install it from [oracle.com](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)

Add following code to your e.g. `.bash_profile`
```bash
# For Apache Spark
if which java > /dev/null; then export JAVA_HOME=$(/usr/libexec/java_home); fi
```

Install Apache Spark
--------------------
You can use Mac OS package manager Brew ([http://brew.sh/](http://brew.sh/))
```shell
brew update
brew install scala
brew install apache-spark
```

Set up env variables
--------------------
Add following code to your e.g. `.bash_profile`
```bash
# For a ipython notebook and pyspark integration
if which pyspark > /dev/null; then
  export SPARK_HOME="/usr/local/Cellar/apache-spark/1.3.1_1/libexec/"
  export PYSPARK_SUBMIT_ARGS="--master local[2]"
fi
```

You can check `SPARK_HOME` path using following brew command
```
$ brew info apache-spark
apache-spark: stable 1.3.1, HEAD
https://spark.apache.org/
/usr/local/Cellar/apache-spark/1.3.1_1 (361 files, 278M) *
  Built from source
From: https://github.com/Homebrew/homebrew/blob/master/Library/Formula/apache-spark.rb
```


Create ipython profile
----------------------
Run
```
ipython profile create pyspark
```

Create a startup file
```
$ vim ~/.ipython/profile_pyspark/startup/00-pyspark-setup.py
```

```python
# Configure the necessary Spark environment
import os
import sys

spark_home = os.environ.get('SPARK_HOME', None)
sys.path.insert(0, spark_home + "/python")

# Add the py4j to the path.
# You may need to change the version number to match your install
sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.8.2.1-src.zip'))

# Initialize PySpark to predefine the SparkContext variable 'sc'
execfile(os.path.join(spark_home, 'python/pyspark/shell.py'))
```

Run ipython
-----------
```
ipython notebook --profile=pyspark
```

`sc` variable should be available
```ipython
In [1]: sc
Out[1]: <pyspark.context.SparkContext at 0x10a982b10>
```