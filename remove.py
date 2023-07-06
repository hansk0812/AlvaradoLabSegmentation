import os
import glob

print (os.path.join("models", os.environ["EXPTNAME"], "channels256/img256/*"))

#
for models in sorted(glob.glob(os.path.join("models", os.environ["EXPTNAME"], "channels256/img256/*")), key=lambda x: int(x.split('.pt')[0].split('epoch')[-1]))[-1:]:
    print (models) 
    os.remove(models)
