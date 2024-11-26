import gradio as gr
import requests
import time
import uuid
import os
from huggingface_hub import HfApi, hf_hub_download
import pandas as pd
import shutil
import json
from pathlib import Path

PAGE_SIZE = 5
FILE_DIR_PATH = "."

repo_id = os.environ["DATASET"]

def append_videos_to_dataset( 
    video_urls,
    video_paths, 
    prompts=None,
    split="train", 
    commit_message="Added new videos"
):
    api = HfApi()
    temp_dir = Path("temp_dataset_folder")
    split_dir = temp_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download existing metadata if it exists
        try:
            metadata_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{split}/metadata.csv",
                repo_type="dataset"
            )
            existing_metadata = pd.read_csv(metadata_path)
            if 'prompt' not in existing_metadata.columns:
                existing_metadata['prompt'] = ''
        except:
            existing_metadata = pd.DataFrame(columns=['file_name', 'prompt'])
        
        # Prepare new metadata entries
        new_entries = []
        for i, video_path in enumerate(video_paths):
            video_name = Path(video_path).name
            
            # Copy video to temporary directory
            shutil.copy2(video_path, split_dir / video_name)
            
            # Add metadata entry with prompt
            new_entries.append({
                'file_name': video_name,
                'prompt': prompts[i] if prompts else '',
                'original_url': video_urls[i] if video_urls else ''
            })
        
        # Combine existing and new metadata
        new_metadata = pd.concat([
            existing_metadata,
            pd.DataFrame(new_entries)
        ]).drop_duplicates(subset=['file_name'], keep='last')
        
        # Ensure no NaN values in prompts
        new_metadata['prompt'] = new_metadata['prompt'].fillna('')
        
        # Save updated metadata
        new_metadata.to_csv(split_dir / 'metadata.csv', index=False)
        
        # Upload to Hugging Face Hub
        api.upload_folder(
            folder_path=str(temp_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message
        )
    
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)



def generate_video(prompt, size, duration, generation_history, progress=gr.Progress()):
    url = 'https://sora.openai.com/backend/video_gen?force_paragen=false'
    
    headers = json.loads(os.environ["HEADERS"])

    cookies = json.loads(os.environ["COOKIES"])
    if size == "1080p":
        width = 1920
        height = 1080
    elif size == "720p":
        width = 1280
        height = 720
    elif size == "480p":
        width = 854
        height = 480
    elif size == "360p":
        width = 640
        height = 360
    payload = {
        "type": "video_gen",
        "prompt": prompt,
        "n_variants": 1,
        "n_frames": 30 * duration,
        "height": height,
        "width": width,
        "style": "natural",
        "inpaint_items": [],
        "model": "turbo",
        "operation": "simple_compose"
    }
    
    # Initial request to generate video
    response = requests.post(url, headers=headers, cookies=cookies, json=payload)
    
    if response.status_code != 200:
        raise gr.Error("Something went wrong")
    
    task_id = response.json()["id"]
    gr.Info("Video generation started. Please wait...")

    # Check status URL
    status_url = 'https://sora.openai.com/backend/video_gen?limit=10'
    
    # Poll for completion
    max_attempts = 60  # Maximum number of attempts
    attempt = 0
    
    while attempt < max_attempts:
        try:
            status_response = requests.get(status_url, headers=headers, cookies=cookies)
            if status_response.status_code == 200:
                list_responses = status_response.json()
                
                for task_response in list_responses["task_responses"]:
                    if task_response["id"] == task_id:
                        print(task_response)
                        if "progress_pct" in task_response:
                            if(task_response["progress_pct"]):
                                progress(task_response["progress_pct"])
                        if "failure_reason" in task_response:
                            if(task_response["failure_reason"]):
                                raise gr.Error(f"Your generation errored due to: {task_response['failure_reason']}")
                        if "moderation_result" in task_response:
                            if(task_response["moderation_result"]):
                                if "is_output_rejection" in task_response["moderation_result"]:
                                    if(task_response["moderation_result"]["is_output_rejection"]):
                                        raise gr.Error(f"Your generation got blocked by OpenAI")
                        if "generations" in task_response:
                            if(task_response["generations"]):
                                print("Generation suceeded")
                                video_url = task_response["generations"][0]["url"]
                                random_uuid = uuid.uuid4().hex
                                unique_filename = f"{FILE_DIR_PATH}/output_{random_uuid}.mp4"
                                unique_textfile = f"{FILE_DIR_PATH}/output_{random_uuid}.txt"
                                video_path, prompt_path = download_video(video_url, prompt, unique_textfile, unique_filename)
                                generation_history = generation_history + ',' + unique_filename
                                append_videos_to_dataset([video_url], [video_path], [prompt])
                                if "actions" in task_response:
                                    if(task_response["actions"]):
                                        generated_prompt = json.dumps(task_response["actions"], sort_keys=True, indent=4)
                                    else:
                                        generated_prompt = None                                        
                                print(generated_prompt)
                                return video_path, generation_history, generated_prompt
            else:
                print(status_response.text)

            time.sleep(5)  # Wait 10 seconds before next attempt
            attempt += 1    
            
        except Exception as e:
            raise gr.Error(f"Error checking status: {str(e)}")
    gr.Error("Timeout: Video generation took too long. Please try again.")

def list_all_outputs(generation_history):
    directory_path = FILE_DIR_PATH
    files_in_directory = os.listdir(directory_path  )
    wav_files = [os.path.join(directory_path, file) for file in files_in_directory if file.endswith('.mp4')]
    wav_files.sort(key=lambda x: os.path.getmtime(os.path.join(directory_path, x)), reverse=True)
    history_list = generation_history.split(',') if generation_history else []
    updated_files = [file for file in wav_files if file not in history_list]
    updated_history = updated_files + history_list
    return ','.join(updated_history)

def increase_list_size(list_size):
    return list_size+PAGE_SIZE

def download_video(url, prompt, save_path_text, save_path_video):
    try:
        # Send a GET request to the URL
        print("Starting download...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path_text, "w") as file:
            file.write(prompt)

        # Open the file in binary write mode
        with open(save_path_video, 'wb') as video_file:
            # Write the content to the file with progress updates
            for chunk in response.iter_content(chunk_size=2 * 1024 * 1024):
                if chunk:
                    video_file.write(chunk)
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the video: {e}")
    except IOError as e:
        print(f"Error saving the file: {e}")
    return save_path_video, save_path_text
css = '''
p, li{font-size: 16px}
code{font-size: 18px}
'''
# Create Gradio interface
with gr.Blocks(css=css) as demo:
    with gr.Tab("Generate with Sora"):
        gr.Markdown("# Sora PR Puppets")
        gr.Markdown("An artists open letter, click on the 'Why are we doing this' tab to learn more")
        generation_history = gr.Textbox(visible=False)
        list_size = gr.Number(value=PAGE_SIZE, visible=False)
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Enter your prompt",
                    placeholder="Describe the video you want to generate...",
                    lines=3
                )
                generate_button = gr.Button("Generate Video")
            with gr.Column():
                output = gr.Video(label="Generated Video")
                generated_prompt = gr.Code(label="Generated prompt", interactive=False, language="json", wrap_lines=True, lines=1)
        with gr.Accordion("Advanced Options", open=True):
            size = gr.Radio(["360p", "480p", "720p", "1080p"], label="Resolution", value="360p", info="Trade off between resolution and speed")
            duration = gr.Slider(minimum=5, maximum=10, step=5, label="Duration", value=10)
        with gr.Accordion("Generation gallery"):
            @gr.render(inputs=[generation_history, list_size])
            def show_output_list(generation_history, list_size):
                metadata_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"train/metadata.csv",
                    repo_type="dataset"
                )
                existing_metadata = pd.read_csv(metadata_path)
                print(existing_metadata)
                for index, generation_list in existing_metadata.iloc[-list_size:][::-1].iterrows():
                    print(generation_list)
                    generation_prompt = generation_list['prompt']
                    generation = generation_list['original_url']                    
                #history_list = generation_history.split(',') if generation_history else []
                #history_list_latest = history_list[:list_size]
                #for generation in history_list_latest:
                #    generation_prompt_file = generation.replace('.mp4', '.txt')
                #    with open(generation_prompt_file, 'r') as file:
                #        generation_prompt = file.read()
                    with gr.Group():
                        gr.Markdown(value=f"### {generation_prompt}")
                        gr.HTML(f'''
                        <video controls width="100%">
                          <source src="{generation}" type="video/mp4" />
                        </video>
                        ''')
            load_more = gr.Button("Load more")
            load_more.click(fn=increase_list_size, inputs=list_size, outputs=list_size)    
    with gr.Tab("Open letter: why are we doing this?"):
        gr.Markdown('''# ┌∩┐(◣_◢)┌∩┐ DEAR CORPORATE AI OVERLORDS ┌∩┐(◣_◢)┌∩┐

We received access to Sora with the promise to be early testers, red teamers and creative partners. However, we believe instead we are being lured into "art washing" to tell the world that Sora is a useful tool for artists. 

<code style="font-family: monospace;font-size: 16px;font-weight:bold">ARTISTS ARE NOT YOUR UNPAID R&D <br />
☠️ we are not your: free bug testers, PR puppets, training data, validation tokens ☠️ </code>

Hundreds of artists provide unpaid labor through bug testing, feedback and experimental work for the program for a $150B valued company. While hundreds contribute for free, a select few will be chosen through a competition to have their Sora-created films screened — offering minimal compensation which pales in comparison to the substantial PR and marketing value OpenAI receives.

<code style="font-family: monospace;font-size: 16px;font-weight:bold">▌║█║▌║█║▌║ DENORMALIZE BILLION DOLLAR BRANDS EXPLOITING ARTISTS FOR UNPAID R&D AND PR ║▌║█║▌║█║▌ </code>

Furthermore, every output needs to be approved by the OpenAI team before sharing. This early access program appears to be less about creative expression and critique, and more about PR and advertisement.

<code style="font-family: monospace;font-size: 16px;font-weight:bold">[̲̅$̲̅(̲̅ )̲̅$̲̅] CORPORATE ARTWASHING DETECTED [̲̅$̲̅(̲̅ )̲̅$̲̅]</code>

We are releasing this tool to give everyone an opportunity to experiment with what ~300 artists were offered: a free and unlimited access to this tool.

We are not against the use of AI technology as a tool for the arts (if we were, we probably wouldn't have been invited to this program). What we don't agree with is how this artist program has been rolled out and how the tool is shaping up ahead of a possible public release. We are sharing this to the world in the hopes that OpenAI becomes more open, more artist friendly and supports the arts beyond PR stunts.

### We call on artists to make use of tools beyond the proprietary:

Open Source video generation tools allow artists to experiment with the avant garde free from gate keeping, commercial interests or serving as PR to any corporation. We also invite artists to train their own models with their own datasets.

Some open source video tools available are:
Open Source video generation tools allow artists to experiment with avant garde tools without gate keeping, commercial interests or serving as a PR to any corporation. Some open source video tools available are:
- [CogVideoX](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce)
- [Mochi 1](https://huggingface.co/genmo/mochi-1-preview)
- [LTX Video](https://huggingface.co/Lightricks/LTX-Video)
- [Pyramid Flow](https://huggingface.co/rain1011/pyramid-flow-miniflux)

However, as we are aware not everyone has the hardware or technical capability to run open source tools and models, we welcome tool makers to listen to and provide a path to true artist expression, with fair compensation to the artists.

Enjoy,

some sora-alpha-artists

''', elem_id="manifesto")
    generate_button.click(
        fn=generate_video,
        inputs=[prompt_input, size, duration, generation_history],
        outputs=[output, generation_history, generated_prompt],
        concurrency_limit=4
    )
    timer = gr.Timer(value=30)
    timer.tick(fn=list_all_outputs, inputs=[generation_history], outputs=[generation_history])
    demo.load(fn=list_all_outputs, inputs=[generation_history], outputs=[generation_history])
    
# Launch the app
if __name__ == "__main__":
    demo.launch(ssr_mode=False)