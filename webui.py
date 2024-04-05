import shutil
import gradio as gr
import os
import webbrowser
import subprocess
import json
import yaml
import argparse
import global_exc_handler
from Log4p.core import *

logger = LogManager().GetLogger('webui')

#########################################################
py_dir=r"python" # SET PYTHON PATH HERE!
#########################################################

current_directory = os.path.dirname(os.path.abspath(__file__))



current_yml=None
def get_status():
    global current_yml
    try:
        cfg = yaml.load(open('config.yml',encoding="utf-8"),Loader=yaml.FullLoader)
        current_yml='当前的训练： '+os.path.basename(cfg["project_name"])+"\n\n以下是配置文件内容：\n\n"
        with open('config.yml', mode="r", encoding="utf-8", errors='ignore') as f:
            current_y=f.read()
            current_yml+=current_y
    except Exception as error:
        current_yml=error

get_status()

def p0_write_yml(project_name,log_interval,validate_step,num_steps,batch_size):
     if project_name=='null'or project_name=='':
         return '请选择！'
     config_path=os.path.join('model',project_name,'config.yml')
     config_yml = yaml.load(open(config_path),Loader=yaml.FullLoader)
     config_yml["train"]["log_interval"] = int(log_interval)
     config_yml["train"]["num_steps"] = int(num_steps)
     config_yml["train"]["validate_step"] = int(validate_step)
     config_yml["train"]["batch_size"] = int(batch_size)
     with open(config_path, 'w', encoding='utf-8') as f:
          yaml.dump(config_yml, f) 
     return 'Success'
     
     


list_project = []  
def refresh_project_list():
    global list_project
    list_project = []  
    for item in os.listdir('datasets'):
       item_path = os.path.join('datasets', item)
       if os.path.isdir(item_path):
          list_project.append(item)
    return (project_name.update(choices=list_project),project_name2.update(choices=list_project),'已刷新下拉列表')



def p0_mkdir(name):
    if name!='':
       try:
         wav_path='datasets'
         if not os.path.exists(wav_path):
             logger.warning('datasets文件夹不存在，正在创建...')
             os.mkdir(wav_path)
         wav_path=os.path.join('datasets',name)
         os.mkdir(wav_path)#datasets/xxx/
         os.mkdir(os.path.join(wav_path,'train'))
         os.mkdir(os.path.join(wav_path,'validate'))
         os.mkdir(os.path.join(wav_path,'testing'))
         path='model'
         path=os.path.join('model',name)
         os.mkdir(path)
         try:
            with open('./model/config.yml', mode="r", encoding="utf-8") as f:
                cfg_yml=yaml.load(f,Loader=yaml.FullLoader)
         except:
            with open('config.yml', mode="r", encoding="utf-8") as f:
                cfg_yml=yaml.load(f,Loader=yaml.FullLoader)
         cfg_yml["project_name"]=name
         with open(os.path.join(path,"config.yml"), 'w', encoding='utf-8') as f:
            yaml.dump(cfg_yml, f)
         refresh_project_list()
         return project_name.update(choices=list_project,value=name),f'Success. 请将数据集按标签放入制定名称文件夹中，并将其写入character.py中。然后进行下一步操作。'
       except Exception as error:
         logger.error(f"发生了一个错误:{error}")
         return error
    else:
       return '请输入名称！'
       
def p0_load_cfg(projectname):
    if projectname=='null'or projectname=='':
        return p0_status.update(value=current_yml),'请选择！'
    try:
        shutil.copy(os.path.join('model',projectname,'config.yml'),'config.yml')
        get_status()
        return p0_status.update(value=current_yml) ,'Success'
    except Exception as error:
        return p0_status.update(value=current_yml),error
        
def a4a_train(project_name):
     command = f"{py_dir} train.py -n {project_name}"
     cfg_path=os.path.join('model',project_name,'config.yml')         
     configjson = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
     if not configjson["train"]["train_countinue"]:
         configjson["train"]["train_countinue"]=False
         with open(cfg_path, 'w', encoding='utf-8') as f:
             json.dump(configjson, f, indent=2, ensure_ascii=False)
         print("已经修改配置文件！\n")
     configyml = yaml.load(open("config.yml"),Loader=yaml.FullLoader)
     configyml["train"]["train_countinue"]=configjson["train"]["train_countinue"]
     subprocess.Popen(['start', 'cmd', '/k', command],cwd=current_directory,shell=True)
     print(command+'\n\n')
     return '已开始训练,关注新窗口信息.关闭窗口或Ctrl+C终止训练'

def a4b_train_cont(project_name):
     command = f"{py_dir} train.py -n {project_name}"
     cfg_path=os.path.join('model',project_name,'config.yml')   
     
     configjson = yaml.load(open(cfg_path,encoding="utf-8"), Loader=yaml.FullLoader)
     if configjson["train"]["train_countinue"]==False:
         configjson["train"]["train_countinue"]=True
         print("已经修改配置文件！\n")
     configyml = yaml.load(open("config.yml",encoding="utf-8"),Loader=yaml.FullLoader)
     configyml["train"]["train_countinue"]=configjson["train"]["train_countinue"]
     subprocess.Popen(['start', 'cmd', '/k', command],cwd=current_directory,shell=True)
     print(command+'\n\n')
     return '已开始训练,关注新窗口信息.关闭窗口或Ctrl+C终止训练'

ckpt_list = ['null']

def c2_refresh_sub_opt(name):  
   try:
       global ckpt_list
       ckpt_list=['null']
       file_list = os.listdir(os.path.join("model",name))
       for ck in file_list:
         if os.path.splitext(ck)[-1] == ".pth":
            ckpt_list.append(ck)
       return models_in_project.update(choices=ckpt_list,value=ckpt_list[-1])
   except :
       return models_in_project.update(choices=['null'],value='null')


def c2_infer(proj_name,model_name,sr,scr_path,js_opt):
    if proj_name=='null' or model_name=='null':
        return '请选择模型！'
        
    path=f'./model/{proj_name}'
    command = f'{py_dir} infrence.py -m {path}/{model_name} -sr {int(sr)} -scr {scr_path} -opt {js_opt}'
    print(command+'\n\n')
    subprocess.Popen(['start', 'cmd', '/k', command],cwd=current_directory,shell=True)
    return '新的命令行窗口已经打开，请关注输出信息。关闭窗口结束推理服务。'




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--server_port", default=7680,type=int,help="server_port"
    )
    args = parser.parse_args()
    with gr.Blocks(title="Genshin-analsts") as app:
        gr.Markdown(value="""
        Genshin-analsts管理器
                    
        严禁将此项目用于一切违反《中华人民共和国宪法》，《中华人民共和国刑法》，《中华人民共和国治安管理处罚法》和《中华人民共和国民法典》之用途。由使用本整合包产生的问题和作者、原作者无关！！！
        """) 
        with gr.Tabs():
            with gr.TabItem("数据集及准备工作"):
            	with gr.Row(): 
                    with gr.Column():
                        p0_mkdir_name=gr.Textbox(label="这将创建实验文件夹，请输入实验名称,不要包含特殊字符和保留字符。",
                        value="",
                        lines=1,
                        interactive=True)
                        p0_mkdir_output_text = gr.Textbox(label="输出信息", placeholder="点击处理按钮",interactive=False)
                        p0_mkdir_btn=gr.Button(value="创建", variant="primary")

                    gr.Markdown(value="<br>")

                    project_name = gr.Dropdown(label="实验文件夹", choices=list_project, value='null'if not list_project else list_project[-1],interactive=True) 

                    with gr.Row():
                            p0_log_interval = gr.Number(label="模型保存间隔", value="5",interactive=True)
                            p0_validate_step = gr.Number(label="准确度验证间隔", value="5",interactive=True)
                            p0_num_steps = gr.Number(label="训练总步数", value="100",interactive=True)
                            p0_batch_size = gr.Number(label="batch_size", value="16",interactive=True)
                            
                    p0_load_cfg_output_text = gr.Textbox(label="输出信息", placeholder="点击处理按钮",interactive=False)
                    with gr.Row():
                            p0_write_cfg_btn=gr.Button(value="保存更改(但不会自动加载)", variant="primary")
                            p0_load_cfg_btn = gr.Button(value="加载训练配置", variant="primary")
                            p0_load_cfg_refresh_btn=gr.Button(value="刷新选项", variant="secondary")
                            
                    with gr.Column():
                        #p0_current_proj=gr.Textbox(label="当前生效的训练",value="",interactive=False)
                        p0_status=gr.TextArea(label="训练前请确认当前的全局配置信息", value=current_yml,interactive=False)
                        
                        
                       
            with gr.TabItem("训练"):
                with gr.Row():              
                    with gr.Row():
                       a4a_btn = gr.Button(value="首次训练", variant="primary")
                       a4b_btn = gr.Button(value="继续训练", variant="primary")                   
                    with gr.Column():
                       a4_textbox_output_text = gr.Textbox(label="输出信息", placeholder="点击处理按钮",interactive=False)
                    

            with gr.TabItem("推理"):
                 gr.Markdown(value='工作区模型推理(model内各实验目录下的模型)')
                 with gr.Row():
                    project_name2 = gr.Dropdown(label="选择实验名", choices=list_project, value='null',interactive=True)
                    models_in_project = gr.Dropdown(label="选择模型", choices=ckpt_list, value='null'if not ckpt_list else ckpt_list[0],interactive=True)
                    
                    with gr.Column():
                        p0_sr = gr.Number(label="采样率", value="44100",interactive=True)
                    with gr.Column():
                        p0_js_opt=gr.Textbox(label="推理结果输出json",
                   	value="./opt.json",
                   	lines=1,
                   	interactive=True)
                    with gr.Column():
                        p0_scr=gr.Textbox(label="需要推理的数据集目录，目录内不得有文件夹",
                   	value="",
                   	lines=1,
                   	interactive=True)
                    
                    with gr.Column():
                       c2_btn = gr.Button(value="启动推理", variant="primary")
                       c2_btn_refresh=gr.Button(value="刷新选项", variant="secondary")
                 with gr.Column():
                       c2_textbox_output_text = gr.Textbox(label="输出信息", placeholder="点击处理按钮",interactive=False)
                 project_name2.change(c2_refresh_sub_opt,[project_name2],[models_in_project])











        p0_write_cfg_btn.click(p0_write_yml,
                           inputs=[project_name,p0_log_interval,p0_validate_step,p0_num_steps,p0_batch_size],
                           outputs=[
                p0_load_cfg_output_text,
            ],)
        
        
        p0_mkdir_btn.click(p0_mkdir,
                           inputs=[p0_mkdir_name],
                           outputs=[
                               project_name,
                p0_mkdir_output_text,
            ],)
        p0_load_cfg_btn.click(p0_load_cfg,
                           inputs=[project_name],
                           outputs=[p0_status,
                p0_load_cfg_output_text,
            ],)
        p0_load_cfg_refresh_btn.click(refresh_project_list,
                           inputs=[],
                           outputs=[project_name,
                                    project_name2,
                p0_load_cfg_output_text,
            ],)
            
            
            
        a4a_btn.click(
            a4a_train,
            inputs=[project_name],
            outputs=[
                a4_textbox_output_text,
            ],
        )

        a4b_btn.click(
            a4b_train_cont,
            inputs=[project_name],
            outputs=[
                a4_textbox_output_text,
            ],
        )
        
        
        c2_btn.click(
            c2_infer,
            inputs=[project_name2,models_in_project,p0_sr,p0_scr,p0_js_opt],
            outputs=[
                c2_textbox_output_text,
            ],
        )
        
        c2_btn_refresh.click(refresh_project_list,[],[project_name,project_name2,c2_textbox_output_text])
                   	
logger.info("正在启动webui")
webbrowser.open(f"http://127.0.0.1:{args.server_port}")
app.launch(share=False,server_port=args.server_port)
