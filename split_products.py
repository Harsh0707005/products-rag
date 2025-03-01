import json

def split_json_file(input_file, num_parts=5):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_items = len(data)
    chunk_size = total_items // num_parts
    remainder = total_items % num_parts
    
    start = 0
    for i in range(num_parts):
        end = start + chunk_size + (1 if i < remainder else 0)
        part_data = data[start:end]
        output_file = f'products{i+1}.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(part_data, f, indent=4)
        
        print(f'Created {output_file} with {len(part_data)} items')
        
        start = end


split_json_file('products.json')
