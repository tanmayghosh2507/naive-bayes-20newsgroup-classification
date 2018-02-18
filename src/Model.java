import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class Model {

	/*
	 * This method builds the model. Both BE and MLE models are written on separate
	 * files.
	 */
	public void buildModel(String trainData, String trainLabel, String modelBE, String modelMLE, String vocabulary)
			throws IOException {
		HashMap<Integer, HashMap<Integer, Integer>> docWordMap = buildDocWordMapping(trainData);
		HashMap<Integer, String> vocabMap = buildVocabMap(vocabulary);
		int vocabSize = vocabMap.size();

		FileWriter fileWriterBE = new FileWriter(modelBE);
		FileWriter fileWriterMLE = new FileWriter(modelMLE);

		HashMap<Integer, HashSet<Integer>> catDocMap = buildCategoryDocMap(trainLabel);
		HashMap<Integer, Integer> catMap;

		for (int category : catDocMap.keySet()) { // For each category
			catMap = new HashMap<Integer, Integer>();
			int n = 0;
			for (int docId : catDocMap.get(category)) { // For each document present under the current category
				for (int wordId : docWordMap.get(docId).keySet()) { // For each word present inside the current document
					catMap.put(wordId, catMap.getOrDefault(wordId, 0) + docWordMap.get(docId).get(wordId)); // Add
																											// frequency
																											// of the
																											// current
																											// word to
																											// catMap.
					n += docWordMap.get(docId).get(wordId); // Calculate total frequency of all words present inside
															// this category.
				}
			}

			double nk = 0.0;
			for (int word : vocabMap.keySet()) {
				nk = catMap.getOrDefault(word, 0);
				// Using formulas of calculating BE and MLE and writing the model to model
				// files.
				fileWriterMLE.write(category + "\t" + word + "\t" + (double) nk / n + "\n");
				fileWriterBE.write(category + "\t" + word + "\t" + (double) (nk + 1) / (n + vocabSize) + "\n");
			}
		}

		fileWriterMLE.close();
		fileWriterBE.close();
	}

	/*
	 * The method takes the label_map, and returns a map with key as document id
	 * and, corresponding value as the category the document belongs to.
	 */
	public HashMap<Integer, Integer> buildLabelMapping(String file) throws IOException {
		HashMap<Integer, Integer> mapping = new HashMap<Integer, Integer>();
		List<String> lines = Files.readAllLines(Paths.get(file), Charset.defaultCharset());

		int docId = 1;
		for (String line : lines) {
			mapping.put(docId, Integer.parseInt(line.trim()));
			docId++;
		}
		return mapping;
	}

	/*
	 * This method takes the train_data/test_data file and returns a map with key as
	 * the document id, and corresponding value as again a map with all the words
	 * contained in the document with it's corresponding frequency count.
	 */
	public HashMap<Integer, HashMap<Integer, Integer>> buildDocWordMapping(String file) throws IOException {
		HashMap<Integer, HashMap<Integer, Integer>> docMap = new HashMap<Integer, HashMap<Integer, Integer>>();
		List<String> lines = Files.readAllLines(Paths.get(file), Charset.defaultCharset());
		int docId;
		for (String str : lines) {
			String[] line = str.split(",");
			docId = Integer.parseInt(line[0]);

			HashMap<Integer, Integer> innerMap = docMap.getOrDefault(docId, new HashMap<Integer, Integer>());
			innerMap.put(Integer.parseInt(line[1]), Integer.parseInt(line[2]));

			docMap.put(docId, innerMap);
		}

		return docMap;
	}

	/*
	 * This method takes the vocabulary file as input and returns a map with key as
	 * the word id and, corresponding value as the word.
	 */
	public HashMap<Integer, String> buildVocabMap(String file) throws IOException {
		HashMap<Integer, String> vocabMap = new HashMap<Integer, String>();
		List<String> lines = Files.readAllLines(Paths.get(file), Charset.defaultCharset());
		int wId = 1;
		for (String line : lines) {
			vocabMap.put(wId, line.trim());
			wId++;
		}
		return vocabMap;
	}

	/*
	 * This method takes the label file, and returns a map with key as group number,
	 * and values as the set of all documents, which are part of the respective
	 * group.
	 */
	public HashMap<Integer, HashSet<Integer>> buildCategoryDocMap(String file) throws IOException {
		HashMap<Integer, HashSet<Integer>> catDocMap = new HashMap<Integer, HashSet<Integer>>();
		List<String> lines = Files.readAllLines(Paths.get(file), Charset.defaultCharset());
		int wId = 1;

		for (String line : lines) {
			HashSet<Integer> list = catDocMap.getOrDefault(Integer.parseInt(line.trim()), new HashSet<Integer>());
			list.add(wId);
			wId++;
			catDocMap.put(Integer.parseInt(line.trim()), list);
		}
		return catDocMap;
	}

	/*
	 * This method takes the map.csv, and returns a mapping of group number and
	 * newsgroup name corresponding group id.
	 */
	public HashMap<Integer, String> buildNewsGroupMapping(String file) throws IOException {
		HashMap<Integer, String> mapping = new HashMap<Integer, String>();
		List<String> lines = Files.readAllLines(Paths.get(file), Charset.defaultCharset());
		for (String line : lines) {
			String[] str = line.split(",");
			mapping.put(Integer.parseInt(str[0]), str[1]);
		}
		return mapping;
	}
}
