import os
import sys
import re

dataset_path = "/DATA1/kchanwo/clipall/datasets/PACS"

if __name__ == '__main__':
    os.makedirs(dataset_path, exist_ok=True)
    domain_list = os.listdir(dataset_path)
    for domain_ in domain_list:
        classes_path = os.path.join(dataset_path, domain_)
        os.makedirs(classes_path, exist_ok=True)
        classes_list = os.listdir(classes_path)
        for class_ in classes_list:
            texts_path = os.path.join(classes_path, class_)
            os.makedirs(texts_path, exist_ok=True)
            texts_list = os.listdir(texts_path)
            for name_ in texts_list:
                if name_[-3:] != 'txt': continue
                UPLOADED_FILE = os.path.join(images_path, name_)
                TO_SAVE_FILE = UPLOADED_FILE[:-3] + 'txt'
                with open(TO_SAVE_FILE, 'w') as f:
                    f.write("") # init file
                image = io.imread(UPLOADED_FILE)
                pil_image = PIL.Image.fromarray(image)
                image = preprocess(pil_image).unsqueeze(0).to(device)
                for class__ in classes_list:
                    prompt = f"It is a {class__}."

                    with torch.no_grad():
                        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
                        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
                    if use_beam_search:
                        generated_text_prefix = generate_beam(model, tokenizer,
                                                            prompt=prompt,
                                                            embed=prefix_embed)[0]
                    else:
                        generated_text_prefix = generate2(model, tokenizer,
                                                        prompt=prompt,
                                                        embed=prefix_embed)
                    generated_text_prefix = generated_text_prefix.strip()
                    if class_ in generated_text_prefix: 
                        generated_text_prefix = generated_text_prefix.replace(class_, class__)
                    # if class__ not in generated_text_prefix and class_ not in generated_text_prefix:
                    if generated_text_prefix[-1] != '.' : generated_text_prefix = generated_text_prefix+'.'
                    generated_text_prefix = prompt +' '+ generated_text_prefix

                    print(f"{domain_}> {class_}> {name_}> {generated_text_prefix}")
                    with open(TO_SAVE_FILE, 'a') as f:
                        f.write(generated_text_prefix+'\n')
                


