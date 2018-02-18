import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class NaiveBayes {

	// Model and prediction files created during the program in the working
	// directory.
	static final String trainPrediction = "train_prediction.tsv";
	static final String testPredictionBE = "test_prediction_be.tsv";
	static final String testPredictionMLE = "test_prediction_mle.tsv";
	static final String modelBE = "model_be.tsv";
	static final String modelMLE = "model_mle.tsv";

	public static void main(String[] args) throws IOException {
		// Check for arguments. Accepts 6.
		if (args.length != 6) {
			System.out.println("You must provide all the arguments.  Please try again.");
		} else {
			String vocabulary = args[0];
			String map = args[1];
			String trainLabel = args[2];
			String trainData = args[3];
			String testLabel = args[4];
			String testData = args[5];

			Model naiveBayes = new Model();
			// Build the model using Train Data, Train Label and Vocabulary.
			// Other arguments are just the file name, to which we want to write the model.
//			naiveBayes.buildModel(trainData, trainLabel, modelBE, modelMLE, vocabulary);

			HashMap<Integer, Integer> trainLabelMap = naiveBayes.buildLabelMapping(trainLabel);
			HashMap<Integer, Integer> testLabelMap = naiveBayes.buildLabelMapping(testLabel);
			HashMap<Integer, String> vocabMap = naiveBayes.buildVocabMap(vocabulary);
			HashMap<Integer, Integer> predictionMap;
			HashMap<Integer, HashSet<Integer>> catDocMapTrain = naiveBayes.buildCategoryDocMap(trainLabel);
			HashMap<Integer, HashSet<Integer>> catDocMapTest = naiveBayes.buildCategoryDocMap(testLabel);

			HashMap<Integer, Double> priorMap = calculateClassPrior(trainLabelMap);
			for (int i : priorMap.keySet())
				System.out.printf("P(Omega = %d) = %.4f \n", i, priorMap.get(i));

			System.out.println("\nTraining Data Statistics(BE): ");
			predictionMap = prediction(vocabMap, priorMap, trainData, modelBE, trainPrediction);
			printStatistics(catDocMapTrain, trainLabelMap, predictionMap);

			System.out.println("\nTest Data Statistics(BE): ");
			predictionMap = prediction(vocabMap, priorMap, testData, modelBE, testPredictionBE);
			printStatistics(catDocMapTest, testLabelMap, predictionMap);

			System.out.println("\nTest Data Statistics(MLE): ");
			predictionMap = prediction(vocabMap, priorMap, testData, modelMLE, testPredictionMLE);
			printStatistics(catDocMapTest, testLabelMap, predictionMap);
		}
	}

	// Calculating Class Priors for each category
	static HashMap<Integer, Double> calculateClassPrior(HashMap<Integer, Integer> labelMap) throws IOException {
		HashMap<Integer, Integer> countMap = new HashMap<Integer, Integer>();
		int count = 0;
		for (int docId : labelMap.keySet()) {
			countMap.put(labelMap.get(docId), countMap.getOrDefault(labelMap.get(docId), 0) + 1);
			count++;
		}

		HashMap<Integer, Double> priorMap = new HashMap<Integer, Double>();
		for (int i : countMap.keySet())
			priorMap.put(i, (double) countMap.get(i) / count);

		return priorMap;
	}

	// Calculating Accuracy & Confusion Matrix Statistics
	static void printStatistics(HashMap<Integer, HashSet<Integer>> catDocMap, HashMap<Integer, Integer> labelMap,
			HashMap<Integer, Integer> predictionMap) throws IOException {
		HashMap<Integer, Integer> matchedMap = new HashMap<Integer, Integer>();
		HashMap<Integer, Integer> completeMap = new HashMap<Integer, Integer>();
		int count = 0, total = 0;
		for (int docId : labelMap.keySet()) {
			completeMap.put(labelMap.get(docId), completeMap.getOrDefault(labelMap.get(docId), 0) + 1);
			total++;
			if (labelMap.get(docId) == predictionMap.get(docId)) {
				matchedMap.put(labelMap.get(docId), matchedMap.getOrDefault(labelMap.get(docId), 0) + 1);
				count++;
			}
		}

		System.out.printf("Overall Accuracy = %.4f", (double) count / total);
		System.out.println("\nClass Accuracy:");
		for (int i : completeMap.keySet())
			System.out.printf("\tGroup %d: %.4f\n", i, (double) matchedMap.get(i) / completeMap.get(i));

		System.out.println("Confusion Matrix:");
		for (int category : catDocMap.keySet()) {
			HashMap<Integer, Integer> innerMap = new HashMap<Integer, Integer>();
			for (int docId : catDocMap.get(category))
				innerMap.put(predictionMap.get(docId), innerMap.getOrDefault(predictionMap.get(docId), 0) + 1);

			for (int i : catDocMap.keySet())
				System.out.print(innerMap.getOrDefault(i, 0) + "\t");
			System.out.println();
		}
	}

	/*
	 * This method makes prediction.
	 */
	static HashMap<Integer, Integer> prediction(HashMap<Integer, String> vocabMap, HashMap<Integer, Double> priorMap,
			String dataFile, String modelFile, String predictionFile) throws IOException {
		List<String> lines = Files.readAllLines(Paths.get(modelFile), Charset.defaultCharset());
		HashMap<Integer, HashMap<Integer, Double>> modelMap = new HashMap<Integer, HashMap<Integer, Double>>();
		for (String line : lines) { // Reading the model file line by line
			String[] columns = line.split("\\t");
			int category = Integer.parseInt(columns[0]);

			HashMap<Integer, Double> innerMap = modelMap.getOrDefault(category, new HashMap<Integer, Double>());
			innerMap.put(Integer.parseInt(columns[1]), Double.parseDouble(columns[2]));

			// Prepare a map out of the model file, with key as category, and value as a map
			// of vocabulary words with probability value.
			modelMap.put(category, innerMap);
		}

		HashMap<Integer, HashMap<Integer, BigDecimal>> predCatMap = new HashMap<Integer, HashMap<Integer, BigDecimal>>();
		lines = Files.readAllLines(Paths.get(dataFile), Charset.defaultCharset());
		int docId;
		for (String line : lines) {// Reading the data file line by line
			String[] columns = line.split(",");
			docId = Integer.parseInt(columns[0]);

			HashMap<Integer, BigDecimal> innerMap = predCatMap.getOrDefault(docId, new HashMap<Integer, BigDecimal>());
			if (vocabMap.containsKey(Integer.parseInt(columns[1]))) {
				for (int category : priorMap.keySet()) { // For all 20 categories, find the product of probability
															// scores and the prior, for each word in the document.
					innerMap.put(category,
							innerMap.getOrDefault(category,
									BigDecimal.valueOf(priorMap.get(category) * Integer.parseInt(columns[2])))
									.multiply(BigDecimal.valueOf(
											modelMap.get(category).getOrDefault(Integer.parseInt(columns[1]), 1.0))));
				}
			}
			predCatMap.put(docId, innerMap);
		}

		FileWriter prediction = new FileWriter(predictionFile);
		HashMap<Integer, Integer> predictionMap = new HashMap<Integer, Integer>();
		for (int document : predCatMap.keySet()) {
			BigDecimal max = BigDecimal.valueOf(-1.0);
			int predictionCategory = 0;
			for (int category : predCatMap.get(document).keySet()) { // Finding maximum product value among all 20
																		// categories, and predicting that category as
																		// final.
				if (max.compareTo(predCatMap.get(document).get(category)) < 0) {
					predictionCategory = category;
					max = predCatMap.get(document).get(category);
				}
			}
			predictionMap.put(document, predictionCategory);
			prediction.write(predictionCategory + "\n"); // Writing predictions to a file.
		}
		prediction.close();
		return predictionMap;
	}
}
