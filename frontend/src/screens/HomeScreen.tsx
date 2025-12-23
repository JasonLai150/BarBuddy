import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  ScrollView,
  Alert,
  ActivityIndicator,
} from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import { LIFT_TYPES, LiftType } from '../types/lift';

export default function HomeScreen() {
  const [selectedLift, setSelectedLift] = useState<LiftType | null>(null);
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [videoFilename, setVideoFilename] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleSelectVideo = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: 'video/*',
      });

      if (result.type === 'success') {
        setVideoUri(result.uri);
        setVideoFilename(result.name);
        Alert.alert('Success', `Selected: ${result.name}`);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to pick video');
      console.error(error);
    }
  };

  const handleUploadLift = async () => {
    if (!selectedLift) {
      Alert.alert('Error', 'Please select a lift type');
      return;
    }

    if (!videoUri) {
      Alert.alert('Error', 'Please select a video');
      return;
    }

    setIsUploading(true);
    try {
      // TODO: Upload to backend
      Alert.alert(
        'Success',
        `${selectedLift} recorded!\nVideo: ${videoFilename}`
      );
      // Reset after successful upload
      setSelectedLift(null);
      setVideoUri(null);
      setVideoFilename(null);
    } catch (error) {
      Alert.alert('Error', 'Failed to upload lift');
      console.error(error);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Track Your Lift</Text>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Select Lift Type</Text>
        <FlatList
          scrollEnabled={false}
          data={LIFT_TYPES}
          keyExtractor={(item) => item}
          renderItem={({ item }) => (
            <TouchableOpacity
              style={[
                styles.liftButton,
                selectedLift === item && styles.liftButtonSelected,
              ]}
              onPress={() => setSelectedLift(item)}
            >
              <Text
                style={[
                  styles.liftButtonText,
                  selectedLift === item && styles.liftButtonTextSelected,
                ]}
              >
                {item}
              </Text>
            </TouchableOpacity>
          )}
        />
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Upload Video</Text>
        <TouchableOpacity
          style={styles.uploadButton}
          onPress={handleSelectVideo}
        >
          <Text style={styles.uploadButtonText}>
            📹 Choose Video from Library
          </Text>
        </TouchableOpacity>

        {videoFilename && (
          <View style={styles.videoSelected}>
            <Text style={styles.videoSelectedText}>✓ {videoFilename}</Text>
          </View>
        )}
      </View>

      <TouchableOpacity
        style={[
          styles.submitButton,
          isUploading && styles.submitButtonDisabled,
        ]}
        onPress={handleUploadLift}
        disabled={isUploading}
      >
        {isUploading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.submitButtonText}>Record Lift</Text>
        )}
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    padding: 16,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 24,
    marginTop: 16,
    color: '#333',
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 12,
    color: '#333',
  },
  liftButton: {
    backgroundColor: '#fff',
    borderRadius: 8,
    paddingVertical: 12,
    paddingHorizontal: 16,
    marginBottom: 8,
    borderWidth: 2,
    borderColor: '#ddd',
  },
  liftButtonSelected: {
    backgroundColor: '#007AFF',
    borderColor: '#007AFF',
  },
  liftButtonText: {
    fontSize: 16,
    fontWeight: '500',
    color: '#333',
  },
  liftButtonTextSelected: {
    color: '#fff',
  },
  uploadButton: {
    backgroundColor: '#fff',
    borderRadius: 8,
    paddingVertical: 16,
    paddingHorizontal: 16,
    borderWidth: 2,
    borderColor: '#ddd',
    borderStyle: 'dashed',
    alignItems: 'center',
  },
  uploadButtonText: {
    fontSize: 16,
    fontWeight: '500',
    color: '#007AFF',
  },
  videoSelected: {
    marginTop: 12,
    backgroundColor: '#e8f5e9',
    borderRadius: 8,
    paddingVertical: 10,
    paddingHorizontal: 12,
  },
  videoSelectedText: {
    fontSize: 14,
    color: '#2e7d32',
    fontWeight: '500',
  },
  submitButton: {
    backgroundColor: '#007AFF',
    borderRadius: 8,
    paddingVertical: 16,
    alignItems: 'center',
    marginBottom: 32,
  },
  submitButtonDisabled: {
    opacity: 0.6,
  },
  submitButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
});
