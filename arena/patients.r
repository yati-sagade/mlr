subject_name <- c("John Doe", "Jane Doe", "Steve Graves")

temperature <- c(98.1, 98.6, 101.4)

flu_status <- c(FALSE, FALSE, TRUE)

gender <- factor(c("MALE", "FEMALE", "MALE"))

blood <- factor(c("O", "AB", "A"), levels=c("A", "B", "AB", "O"))

pt_data <- data.frame(subject_name, temperature, flu_status, gender, blood,
                      stringsAsFactors=FALSE)

