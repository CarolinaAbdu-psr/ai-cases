import psr.factory
import os
case_path = r"C:\PSR\SDDP18.1Beta\examples\operation\1_stage\Case01"
new_case_path = r"C:\PSR\SDDP18.1Beta\examples\operation\1_stage\Case01-edited"

study = psr.factory.load_study(case_path)


t = study.find("ThermalPlant")[0]
t.set("GerMin",8)
t.set("GerMax", 16)

d_sg = study.find("DemandSegment")[0]
df = d_sg.get_df("EnergyPerBlock")*2
d_sg.set_df(df)

os.makedirs(new_case_path,exist_ok=True)
study.save(new_case_path)

s2 = psr.factory.load_study(new_case_path)
t = s2.find("ThermalPlant")[0]
