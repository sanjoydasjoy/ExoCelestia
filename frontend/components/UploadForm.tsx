import { useState, useRef, ChangeEvent } from 'react';
import styles from '../styles/UploadForm.module.css';

interface UploadFormProps {
  onFileUpload: (file: File) => void;
  isLoading: boolean;
}

export default function UploadForm({ onFileUpload, isLoading }: UploadFormProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      
      // TODO: Add file validation (size, type, format)
      if (
        file.type !== 'text/csv' &&
        file.type !== 'application/json' &&
        !file.name.endsWith('.csv') &&
        !file.name.endsWith('.json')
      ) {
        alert('Please upload a CSV or JSON file');
        return;
      }
      
      setSelectedFile(file);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      
      if (
        file.type !== 'text/csv' &&
        file.type !== 'application/json' &&
        !file.name.endsWith('.csv') &&
        !file.name.endsWith('.json')
      ) {
        alert('Please upload a CSV or JSON file');
        return;
      }
      
      setSelectedFile(file);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (selectedFile) {
      onFileUpload(selectedFile);
    }
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <form onSubmit={handleSubmit} className={styles.uploadForm}>
      <div
        className={`${styles.dropzone} ${dragActive ? styles.dragActive : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.json,application/json,text/csv"
          onChange={handleFileChange}
          className={styles.fileInput}
        />
        
        <div className={styles.dropzoneContent}>
          <div className={styles.uploadIcon}>📁</div>
          
          {selectedFile ? (
            <div className={styles.fileInfo}>
              <p className={styles.fileName}>{selectedFile.name}</p>
              <p className={styles.fileSize}>
                {(selectedFile.size / 1024).toFixed(2)} KB
              </p>
            </div>
          ) : (
            <div>
              <p className={styles.dropzoneText}>
                Drag and drop your CSV/JSON file here, or
              </p>
              <button
                type="button"
                onClick={handleButtonClick}
                className={styles.browseButton}
              >
                Browse Files
              </button>
            </div>
          )}
        </div>
      </div>

      <div className={styles.formatInfo}>
        <h4>Expected CSV Format:</h4>
        <ul>
          <li>Header row with feature names</li>
          <li>Light curve flux values (FLUX.1, FLUX.2, ...)</li>
          <li>Or aggregated features from Kepler/K2/TESS data</li>
        </ul>
        {/* TODO: Add link to example CSV or data format documentation */}
      </div>

      {selectedFile && (
        <button
          type="submit"
          disabled={isLoading}
          className={styles.submitButton}
        >
          {isLoading ? (
            <>
              <span className={styles.spinner}></span>
              Processing...
            </>
          ) : (
            'Analyze for Exoplanets'
          )}
        </button>
      )}
    </form>
  );
}

