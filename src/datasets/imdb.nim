# Copyright 2018 the Arraymancer contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import  httpclient, zip/gzipfiles, strformat, os, strutils,
        ./util, ../tensor/tensor,
        untar

let
    folder_name = "aclImdb"
    file_name = fmt"{folder_name}_v1.tar.gz"

type imdb = tuple[
    train_texts: Tensor[string],
    test_texts: Tensor[string],
    train_labels: Tensor[int],
    test_labels: Tensor[int],
]

template debug(msg: string) =
    when defined(debug):
        echo msg

proc download_imdb_tgz(cache_dir: string) =
    let
        client = newHttpClient()
        imdb_tgz_url = fmt"http://ai.stanford.edu/~amaas/data/sentiment/{file_name}"
        path = cache_dir / file_name
        folder = cache_dir / folder_name

    discard existsOrCreateDir(folder)

    debug(fmt"Downloading {path}")
    client.downloadFile(imdb_tgz_url, path)
    debug("Done!")

proc extract_and_delete_tgz(cache_dir, file_name: string) =
    let tgz = cache_dir / file_name

    debug(fmt"Extracting {tgz}")
    newTarFile(tgz).extract(cache_dir / foldername)
    debug("Done!")
    os.removeFile(tgz)

proc read_imdb(path: string): imdb =
    let section_length = 12_500
    for validation_split in @["train", "test"]:
        var
            texts = newSeq[string](section_length)
            labels = newSeq[int](section_length)

        for sentiment in @["pos", "neg"]:
            let section = path / validation_split / sentiment
            var i = 0
            debug(fmt"Reading section: {section}")
            for file_path in walkFiles(section / "*.txt"):
                # Get the file contexts
                let text = readFile(file_path)
                texts[i] = text.string

                # Extract the label from the filen ame
                let
                    (_, file_name, _) = splitFile(file_path)
                    label = file_name.split("_")[1]
                labels[i] = parseInt(label)
                i += 1

            if validation_split == "train":
                result.train_texts = texts.toTensor()
                result.train_labels = labels.toTensor()
            else:
                result.test_texts = texts.toTensor()
                result.test_labels = labels.toTensor()


proc load_imdb*(cache = true): imdb =
    let cache_dir = get_cache_dir()

    if not dirExists( cache_dir / folder_name):
        create_cache_dirs_if_necessary()
        download_imdb_tgz(cache_dir)
        extract_and_delete_tgz(cache_dir, file_name)

    result = read_imdb(cache_dir / folder_name)
