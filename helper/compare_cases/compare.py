#version 1.6

import psr.factory 
import pandas as pd
import os 
import shutil
import sys
import logging
from pathlib import Path



# Configure logging: file receives all differences (INFO+), console only WARNING+
logger = logging.getLogger("compare_cases")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    # File handler logs everything (INFO and above)
    fh = logging.FileHandler("log.txt", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Console handler only shows warnings and errors
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch_formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    logger.propagate = False

def create_dataframe(obj_key,options, property,date, value_a,value_b):
    """Creates a dataframe with the collumns from the instances"""
    
    data = {'obj_key':[obj_key],
        'op': [options],
        'property': [property],
        'date': [date],
        'value_a':[value_a],
        'value_b':[value_b]}
    df = pd.DataFrame(data)

    df_multi_index = df.set_index(['obj_key', 'op'])

    return df_multi_index

def add_to_dataframe(type, obj_key, options, property,date, value_a, value_b, dataframes):
    """Add a new row to a dataframe of selected object"""

    if type not in dataframes:
        dataframes[type] = create_dataframe(obj_key, options, property,date, value_a,value_b)
    else: 
        df = dataframes[type]
        new_line = create_dataframe(obj_key,options,property,date, value_a,value_b)
        dataframes[type] = pd.concat([df,new_line], ignore_index = False)

    # Log the difference to the file logger (info level)
    try:
        logger.info("Difference added - type=%s obj_key=%s op=%s property=%s date=%s value_a=%s value_b=%s",
                    type, obj_key, options, property, date, value_a, value_b)
    except Exception:
        # If logging is not configured for some reason, silently continue
        pass

    return dataframes

def compare_references(ref_list_source, ref_list_target):
    "Return if two references obejcts are the same"
    for item_source in ref_list_source:
        match = False 
        for item_target in ref_list_target: 
            match =  item_source.key == item_target.key 
            if match: 
                break 
        if not match: # If the item_source are not in ref target list, the objects re different 
            return False
    return match

def normalize_references(obj, key):
    """Normalize references to always be a list"""
    ref_value = obj.get(key)
    # Empty reference
    if ref_value is None:
        return []
    # Refernce is not a list
    if not isinstance(ref_value, list):
        return [ref_value]
    return ref_value

def compare_static_values(obj_source, obj_target, key, dataframes):
    """Compare static properties of two objects"""

    try:
        obj_type = obj_source.type 
        obj_key = obj_source.key
    except:
        obj_type = "Study Object"
        obj_key = obj_source.key
    
    value_a = None
    value_b = None
    try: 
        value_a = obj_source.get(key)
        value_b = obj_target.get(key)
    except Exception as e:
        logger.error('Error getting value for %s: %s', key, e)
        pass 
        
    if value_a != value_b: 
        logger.info("Static difference - %s: %s vs %s", key, value_a, value_b)
        # Register modification of static values
        dataframes = add_to_dataframe(obj_type, obj_key, "M", key, "01/01/1900 ", value_a, value_b, dataframes)
        
    return dataframes


def compare_dynamic_values(obj_source, obj_target, key, dataframes):
    """Compare dynamic properties (dataframes) such as Thermal Plants Installed Capacity modifications"""

    try:
        obj_type = obj_source.type 
        obj_key = obj_source.key
    except:
        obj_type = "Study Object"
        obj_key = obj_source.key
    
    try:
        #Compare two dataframes
        df_a = obj_source.get_df(key)
        df_b = obj_target.get_df(key)
        
        cols1 = set(df_a.columns)
        cols2 = set(df_b.columns)
        comum_cols = cols1.intersection(cols2)

        dif_12 = sorted(cols1 - cols2)
        dif_21 = sorted(cols2 - cols1)

        # Colunms of dataframe a that aren't in dataframe b
        for column_name in dif_12:
            dataframes = add_to_dataframe(obj_type, obj_key, "M", column_name, "-", "Column only in A", "None", dataframes)
        
        # Colunms of dataframe b that aren't in dataframe a
        for column_name in dif_21:
            dataframes = add_to_dataframe(obj_type, obj_key, "M", column_name, "-", "None", "Column only in B", dataframes)

        #Comuns columns 
        for column_name in comum_cols:
            df_compare = df_a[[column_name]].compare(df_b[[column_name]])
            for index, row in df_compare.iterrows(): 
                date = str(index)
                value_a = row[(column_name,"self")]
                value_b = row[(column_name,"other")]
                dataframes = add_to_dataframe(obj_type, obj_key, "M", column_name, date, value_a, value_b, dataframes)
            

    except Exception as e:
        logger.error("Error comparing dynamic value %s: %s", key, e)

        return dataframes

    return dataframes


def is_dataframe(obj,key): 
    """Check if the property is a dataframe"""
    description = obj.description(key)
    if description.is_dynamic():
        return True
    if len(description.dimensions()) > 0: 
        return True
    return False

def compare_study_object(obj_source,obj_target,dataframes):
    """Compare properties from study object"""
    # Get all properties of obj_source
    for key in obj_source.descriptions().keys():

        try:
            # Define object type, code, and name, with exeception for study object
            type = "Study Object"

            #Compare if the properties are equal (static and dynamic)
            description = obj_target.description(key)

            if description is not None: 
                if not  is_dataframe(obj_target,key): 
                    dataframes = compare_static_values(obj_source, obj_target, key, dataframes)

                else: 
                    dataframes = compare_dynamic_values(obj_source, obj_target, key, dataframes)
        except Exception as e:
            logger.exception("Error comparing study object property %s: %s", key, e)
            continue

    return dataframes

def compare_objects(obj_source, obj_target, dataframes): 
    """Compare two objects property by property"""

    # Get all properties of obj_source
    for key in obj_source.descriptions().keys():
        try:
            # Define object type and obj_key, with exeception for study object
            try:
                type = obj_source.type 
                obj_key = obj_source.key
            except:
                type = "Study Object"
                obj_key = obj_source.key
            
            if obj_source.description(key).is_reference() and key.startswith("Ref") and type!= "Study Object": 

                # Normalize objects to always be a list 
                ref_list_source = normalize_references(obj_source, key)
                ref_list_target = normalize_references(obj_target, key)

                # If reference is empty 
                if not ref_list_source and not ref_list_target:
                    continue
                    
                # Call compare_references function to check if the objects are the same
                match = compare_references(ref_list_source, ref_list_target)

                # If refences objects are not the same, add modification on dataframe
                if not match: 
                    logger.info("Different references found for %s obj_key%s key=%s", type, obj_key, key)
                    dataframes = add_to_dataframe(type, obj_key, "M", key, "",ref_list_source, ref_list_target, dataframes)
                continue 

            #Compare if the other properties are equal (static and dynamic)
            description = obj_target.description(key)
            
            if description is not None: 
                if not is_dataframe(obj_target,key): 
                    dataframes = compare_static_values(obj_source, obj_target, key, dataframes)

                else: 
                    dataframes = compare_dynamic_values(obj_source, obj_target, key, dataframes)
        except Exception as e:
            # Log the error but continue with next property
            try:
                logger.exception("Error comparing object %s key %s: %s", obj_key if 'obj_key' in locals() else 'Unknown', key, e)
            except Exception:
                # Fallback simple log
                logger.error("Error comparing object property: %s", e)
            continue

    return dataframes


def compare_studies(study_a, study_b, dataframes={}):
    """Compare two studies and return a dictionary of dataframes"""

    # Compare all objects from study_a with objcts from study_b
    all_objects=study_a.get_all_objects()
    study_b_visited_objects  =[]
    for obj in all_objects:
        try:
            try:
                type = obj.type 
                obj_key = obj.key 
            except:
                type = "Study Object"
                obj_key = obj.key 
            
            #Get correspondet in study B 
            obj_b = study_b.get_by_key(obj_key)
          
            #No correspontet object in study B
            if not obj_b:  
                dataframes= add_to_dataframe(type, obj_key, "R", "None","None","None", "None", dataframes)
            
            # Correspontent in study b 
            else:
                dataframes = compare_objects(obj,obj_b,dataframes)
                study_b_visited_objects.append(obj_b)
        except Exception as e:
            logger.exception("Error comparing object in study %s: %s", obj_key if 'obj_key' in locals() else 'Unknown', e)
            continue
        

    #Get remaining objects of study b: 
    all_objects=study_b.get_all_objects()
    for obj in all_objects:
        try:
            if obj in study_b_visited_objects:
                continue
            
            type = obj.type 
            obj_key = obj.key

            dataframes= add_to_dataframe(type, obj_key, "A", "None","None","None", "None", dataframes)
        except Exception as e:
            logger.exception("Error processing remaining object in study B: %s", e)
            continue

    #Compare study object
    logger.info("Comparing Study Object")
    dataframes = compare_study_object(study_a,study_b, dataframes)

    return dataframes


def save__dataframes(differences):

    output_dir = ".\\helper\\compare_cases\\comparison_results"
    os.makedirs(output_dir, exist_ok=True)

    for filename, df in differences.items():
        try:
            df = df.fillna("None")
            csv_name = f"{filename}.csv"
            path = os.path.join(output_dir, csv_name)

            # Save dataframe to CSV
            df.to_csv(path, index=True)
            logger.info("Saved %s", path)
        except Exception as e:
            logger.exception("Error saving dataframe %s: %s", filename, e)
            continue


def clean_outputs():
    BASE_DIR = Path(__file__).resolve().parent
    output_dir = BASE_DIR / "comparison_results" 

    # Remove existents results
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)  
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  
            except Exception as e:
                logger.exception("Error removing output item %s: %s", item_path, e)
                continue

        logger.info("Previous results from '%s' were deleted", output_dir)


def compare_cases(study_path_a, study_path_b):
    try:
        #Load studies
        study_a = psr.factory.load_study(study_path_a)
        try:
            study_a.save(study_path_a)
        except Exception:
            # non-fatal; continue even if save fails
            logger.exception("Error saving study A to %s", study_path_a)

        study_b = psr.factory.load_study(study_path_b)
        try:
            study_b.save(study_path_b)
        except Exception:
            logger.exception("Error saving study B to %s", study_path_b)

        differences = compare_studies(study_a,study_b)
        try:
            clean_outputs()
        except Exception:
            logger.exception("Error cleaning outputs before saving differences")

        save__dataframes(differences)
    except Exception as e:
        logger.exception("Fatal error comparing cases: %s", e)
        return


if __name__== "__main__":
    a = r"C:\PSR\SDDP18.1Beta\examples\operation\1_stage\Case01"
    b = r"C:\PSR\SDDP18.1Beta\examples\operation\1_stage\Case01-edited"
    compare_cases(a,b)
