"""
OAI-PMH Data Harvester module.

Uses Sickle library with cloudscraper override for safe harvesting
from the Widyatama University repository.
"""

import time
import logging
from pathlib import Path
from typing import Iterator, Optional, Any
from dataclasses import dataclass, field

import pandas as pd
import cloudscraper
from sickle import Sickle
from sickle.models import Record
from tqdm import tqdm

from src.config import Settings, get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetadataRecord:
    """Structured representation of a harvested metadata record."""
    
    identifier: str
    title: str = ""
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    date: str = ""
    subjects: list[str] = field(default_factory=list)
    publisher: str = ""
    types: list[str] = field(default_factory=list)
    language: str = ""
    source: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            "identifier": self.identifier,
            "title": self.title,
            "abstract": self.abstract,
            "authors": "; ".join(self.authors) if self.authors else "",
            "date": self.date,
            "subjects": "; ".join(self.subjects) if self.subjects else "",
            "publisher": self.publisher,
            "types": "; ".join(self.types) if self.types else "",
            "language": self.language,
            "source": self.source,
        }


class SafeSickle(Sickle):
    """
    Extended Sickle class that uses cloudscraper for requests.
    
    This helps bypass common anti-bot protections on repository sites.
    """
    
    def __init__(self, **kwargs) -> None:
        """Initialize SafeSickle with cloudscraper session."""
        super().__init__(**kwargs)
        # Replace the default session with cloudscraper
        self._scraper = cloudscraper.create_scraper()
    
    def _request(self, kwargs):
        """
        Override the default request method to use cloudscraper.
        
        Args:
            kwargs: Request parameters
            
        Returns:
            Response content as bytes
        """
        if self.http_method == 'GET':
            return self._scraper.get(self.endpoint, params=kwargs, **self.request_args)
        return self._scraper.post(self.endpoint, data=kwargs, **self.request_args)


class OAIPMHHarvester:
    """
    Harvester for OAI-PMH compliant repositories.
    
    Uses SafeSickle for safe harvesting with automatic resumption
    token handling and progress tracking.
    """
    
    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Initialize the harvester.
        
        Args:
            settings: Project settings. Uses default if not provided.
        """
        self.settings = settings or get_settings()
        self.sickle = SafeSickle(
            endpoint=self.settings.oaipmh_endpoint,
            max_retries=3,
            default_retry_after=10,
            retry_status_codes=[403, 429, 500, 502, 503, 504],
        )
        self._record_count = 0
    
    def _parse_record(self, record: Record) -> MetadataRecord:
        """
        Parse a Sickle Record into a MetadataRecord.
        
        Args:
            record: Raw Sickle record
            
        Returns:
            Parsed metadata record
        """
        metadata = record.metadata
        
        # Helper to safely get first element or empty string
        def get_first(key: str, default: str = "") -> str:
            values = metadata.get(key, [])
            return values[0] if values else default
        
        # Helper to get all values as list
        def get_all(key: str) -> list[str]:
            return metadata.get(key, [])
        
        return MetadataRecord(
            identifier=record.header.identifier,
            title=get_first("title"),
            abstract=get_first("description"),
            authors=get_all("creator"),
            date=get_first("date"),
            subjects=get_all("subject"),
            publisher=get_first("publisher"),
            types=get_all("type"),
            language=get_first("language"),
            source=get_first("source"),
        )
    
    def harvest(
        self,
        set_spec: Optional[str] = None,
        from_date: Optional[str] = None,
        until_date: Optional[str] = None,
        max_records: Optional[int] = None,
    ) -> Iterator[MetadataRecord]:
        """
        Harvest metadata records from the repository.
        
        Args:
            set_spec: Optional OAI set specification to filter
            from_date: Start date (YYYY-MM-DD format)
            until_date: End date (YYYY-MM-DD format)
            max_records: Maximum number of records to harvest
            
        Yields:
            MetadataRecord objects
        """
        params: dict[str, str] = {
            "metadataPrefix": self.settings.oaipmh_metadata_prefix
        }
        
        if set_spec:
            params["set"] = set_spec
        if from_date:
            params["from"] = from_date
        if until_date:
            params["until"] = until_date
        
        logger.info(f"Starting harvest from {self.settings.oaipmh_endpoint}")
        logger.info(f"Parameters: {params}")
        
        self._record_count = 0
        
        try:
            records = self.sickle.ListRecords(**params)
            
            for record in records:
                # Skip deleted records
                if record.header.deleted:
                    continue
                
                try:
                    parsed = self._parse_record(record)
                    self._record_count += 1
                    
                    if max_records and self._record_count >= max_records:
                        logger.info(f"Reached max_records limit: {max_records}")
                        break
                    
                    yield parsed
                    
                    # Add delay to be nice to the server
                    time.sleep(self.settings.oaipmh_delay_seconds / 10)
                    
                except Exception as e:
                    logger.warning(f"Error parsing record: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Harvest error: {e}")
            raise
    
    def harvest_to_dataframe(
        self,
        set_spec: Optional[str] = None,
        from_date: Optional[str] = None,
        until_date: Optional[str] = None,
        max_records: Optional[int] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Harvest metadata and return as a DataFrame.
        
        Args:
            set_spec: Optional OAI set specification
            from_date: Start date filter
            until_date: End date filter
            max_records: Maximum records to harvest
            show_progress: Show progress bar
            
        Returns:
            DataFrame with harvested metadata
        """
        records_data: list[dict[str, Any]] = []
        
        harvest_iter = self.harvest(
            set_spec=set_spec,
            from_date=from_date,
            until_date=until_date,
            max_records=max_records,
        )
        
        if show_progress:
            harvest_iter = tqdm(
                harvest_iter,
                desc="Harvesting records",
                unit="records",
                total=max_records,
            )
        
        for record in harvest_iter:
            records_data.append(record.to_dict())
        
        logger.info(f"Harvested {len(records_data)} records")
        
        return pd.DataFrame(records_data)
    
    def harvest_and_save(
        self,
        output_path: Optional[Path] = None,
        set_spec: Optional[str] = None,
        from_date: Optional[str] = None,
        until_date: Optional[str] = None,
        max_records: Optional[int] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Harvest metadata and save to CSV file.
        
        Args:
            output_path: Path to save CSV. Uses default if not provided.
            set_spec: Optional OAI set specification
            from_date: Start date filter
            until_date: End date filter
            max_records: Maximum records to harvest
            show_progress: Show progress bar
            
        Returns:
            DataFrame with harvested metadata
        """
        if output_path is None:
            output_path = self.settings.raw_data_dir / self.settings.raw_metadata_file
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = self.harvest_to_dataframe(
            set_spec=set_spec,
            from_date=from_date,
            until_date=until_date,
            max_records=max_records,
            show_progress=show_progress,
        )
        
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Saved {len(df)} records to {output_path}")
        
        return df
    
    def list_sets(self) -> list[dict[str, str]]:
        """
        List available OAI sets from the repository.
        
        Returns:
            List of sets with their specs and names
        """
        sets = []
        try:
            for oai_set in self.sickle.ListSets():
                sets.append({
                    "setSpec": oai_set.setSpec,
                    "setName": oai_set.setName,
                })
        except Exception as e:
            logger.error(f"Error listing sets: {e}")
        
        return sets
    
    def identify(self) -> dict[str, Any]:
        """
        Get repository identification information.
        
        Returns:
            Dictionary with repository information
        """
        try:
            identify = self.sickle.Identify()
            return {
                "repositoryName": identify.repositoryName,
                "baseURL": identify.baseURL,
                "protocolVersion": identify.protocolVersion,
                "adminEmail": identify.adminEmail,
                "earliestDatestamp": identify.earliestDatestamp,
                "deletedRecord": identify.deletedRecord,
                "granularity": identify.granularity,
            }
        except Exception as e:
            logger.error(f"Error identifying repository: {e}")
            return {}
