using EzXML
using Images
function WaterInfo(jpg_path::String)
    file_name1 = split(jpg_path, "/")[end]
    file_name2 = split(file_name1, "\\")[end]
    file_prefix = split(file_name2, ".")[1]
    file_info = split(file_prefix, "_")
    turbidity_string = replace(file_info[1], "ntu" => "")
    turbidity = parse(Float32, turbidity_string)
    temperature = file_info[2]
    light = file_info[3]
    return(turbidity, temperature, light)
end
function Xml2Box(xml_path::String)
    doc = readxml(xml_path)
    xml_root = doc.root
    bndbox = findfirst("//bndbox", xml_root)
    if bndbox !== nothing
        xmin = parse(Int, findfirst("xmin", bndbox).content)
        ymin = parse(Int, findfirst("ymin", bndbox).content)
        xmax = parse(Int, findfirst("xmax", bndbox).content)
        ymax = parse(Int, findfirst("ymax", bndbox).content)
        return (xmin, ymin, xmax, ymax)
    else
        error("No bndbox found in XML")
    end
end
function ImageArray(jpg_path::String, xml_path::String)
    img = load(jpg_path)
    xmin, ymin, xmax, ymax = Xml2Box(xml_path)
    cropped_img = img[ymin:ymax, xmin:xmax]
    new_img = imresize(cropped_img, (224, 224))
    img_array_raw = Float32.(channelview(new_img))
    img_array_one = permutedims(img_array_raw, [3, 2, 1])
    img_array = reshape(img_array_one, (size(img_array_one)[1], size(img_array_one)[2], size(img_array_one)[3], 1))
    return img_array
end
function BatchData(;train_folder::String, id_array::Array)
    train_files = readdir(train_folder)
    train_jpg = sort(filter(x -> endswith(x, ".jpg"), train_files))
    train_xml = sort(filter(x -> endswith(x, ".xml"), train_files))
    batch_size = length(id_array)
    image_decoder_list = []
    turbidity_list = []
    for i in 1:batch_size
        id = id_array[i]
        jpg_path = joinpath(train_folder, train_jpg[id])
        xml_path = joinpath(train_folder, train_xml[id])
        image_decoder = ImageArray(jpg_path, xml_path)
        turbidity, temperature, light = WaterInfo(jpg_path)
        push!(image_decoder_list, image_decoder)
        push!(turbidity_list, turbidity)
    end
    input_decoder = cat(image_decoder_list..., dims=4)
    turbidity_y = Float32.(reshape(turbidity_list, (1,batch_size)))
    data = (input_decoder,turbidity_y)
    return data
end
function CreateBatches(train_num::Int, batch_size::Int)
    total_array = collect(1:train_num)
    shuffled_array = shuffle(total_array)
    num_batches = ceil(Int, train_num/batch_size)
    batches = Vector{Vector{Int}}(undef, num_batches)
    for i in 1:num_batches
        start_idx = (i - 1) * batch_size + 1
        end_idx = i * batch_size
        batches[i] = shuffled_array[start_idx:end_idx]
    end
    return batches
end
